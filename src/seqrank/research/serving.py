"""Versioned batch inference; the caller supplies only completed positive history."""
from __future__ import annotations

import sqlite3
import time
from pathlib import Path
from typing import Annotated

import numpy as np
import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field, model_validator

from .config import file_hash, write_json
from .dataset import load_artifacts
from .evaluation import checkpoint_path, rerank
from .features import dense_features, transform_batch
from .retrievers import ANN, exact_topk
from .training import load_checkpoint


class HistoryEvent(BaseModel):
    model_config = ConfigDict(extra='forbid',allow_inf_nan=False)
    item_id: str
    timestamp: int = Field(ge=0)
    available_at: int = Field(ge=0)
    watch: float = Field(default=-1,ge=-1)
    duration: float = Field(default=-1,ge=-1)
    positive: bool = True

    @model_validator(mode='after')
    def validate_feedback(self):
        if not self.positive or self.available_at<=self.timestamp:
            raise ValueError('History requires primary-positive events with completed feedback')
        if (self.watch<0 and self.watch!=-1) or (self.duration<0 and self.duration!=-1):
            raise ValueError('Missing duration/watch uses -1 exactly')
        return self


class Request(BaseModel):
    model_config = ConfigDict(extra='forbid')
    user_id: str
    timestamp: int = Field(ge=0)
    dataset_version: str
    history: list[HistoryEvent] = Field(default_factory=list,max_length=10000)
    k: int = Field(default=10,ge=1,le=1000)
    candidates: int = Field(default=100,ge=1,le=5000)

    @model_validator(mode='after')
    def validate_time(self):
        if self.k>self.candidates:
            raise ValueError('k cannot exceed candidates')
        if any(h.timestamp>=self.timestamp or h.available_at>=self.timestamp for h in self.history):
            raise ValueError('Every history event and its feedback must be strictly before request time')
        return self


class PredictRequest(Request):
    item_id: str
    duration: float = Field(default=-1,ge=-1,allow_inf_nan=False)
    category: str = 'unknown'


class Bundle:
    def __init__(self,config):
        self.config = config
        self.root = Path(config['data']['prepared'])
        self.manifest,self.statistics,self.catalog = load_artifacts(self.root)
        self.version = self.manifest['fingerprint']
        self.device = torch.device(config.get('serving',{}).get('device','cpu'))
        self.retriever,self.retrieval_state = load_checkpoint(checkpoint_path(config,'retrieval'),self.device,self.version)
        self.ranker,self.ranking_state = load_checkpoint(checkpoint_path(config,'ranking'),self.device,self.version)
        if self.retrieval_state['stage']!='retrieval' or self.ranking_state['stage']!='ranking':
            raise ValueError('Incorrect checkpoint stages')
        self.model_versions = {stage:file_hash(checkpoint_path(config,stage)) for stage in ('retrieval','ranking')}
        ann_path = config.get('serving',{}).get('ann')
        self.ann = ANN(ann_path,self.model_versions['retrieval'],self.version,config.get('ann',{}).get('ef_search',128)) if ann_path else None

    def batch(self,requests):
        result = {k:[] for k in ('user','history','history_watch','history_duration','history_time','timestamp')}
        # Open read-only per call; no shared SQLite connection between FastAPI worker threads.
        with sqlite3.connect(f'file:{self.root / "events.db"}?mode=ro',uri=True) as db:
            def lookup(table,value):
                row = db.execute(f'SELECT id FROM {table} WHERE raw=?',(value,)).fetchone()
                return row[0] if row else 1
            for request in requests:
                if request.dataset_version!=self.version:
                    raise ValueError('Request dataset_version mismatch')
                horizon = min(request.timestamp,self.manifest['train_end']) if self.manifest['history_protocol']=='frozen_train' else request.timestamp
                if any(h.available_at>=horizon for h in request.history):
                    raise ValueError('History violates frozen_train availability cutoff')
                events = sorted(request.history,key=lambda x:x.timestamp)[-self.manifest['max_len']:]
                pad = self.manifest['max_len']-len(events)
                result['user'].append(lookup('user_vocab',request.user_id))
                result['timestamp'].append(request.timestamp)
                for name,values,fill in [('history',[lookup('item_vocab',h.item_id) for h in events],0),
                                         ('history_watch',[h.watch for h in events],-1),
                                         ('history_duration',[h.duration for h in events],-1),
                                         ('history_time',[h.timestamp for h in events],0)]:
                    result[name].append(values+[fill]*pad)
        return {k:torch.tensor(v,device=self.device,dtype=torch.float32 if k in {'history_watch','history_duration'} else torch.long) for k,v in result.items()}

    @torch.inference_mode()
    def recommend(self,requests):
        if not requests:
            raise ValueError('Empty request batch')
        batch = self.batch(requests)
        candidates = max(r.candidates for r in requests)
        exclude = self.config.get('evaluation',{}).get('exclude_seen',False)
        if self.ann:
            if exclude:
                raise ValueError('ANN serving with exclude_seen is unsupported; use exact retrieval')
            _,array = self.ann.search(self.retriever.user_vectors(batch['history']).cpu().numpy(),candidates)
            ids = torch.from_numpy(array).to(self.device)
        else:
            _,ids = exact_topk(self.retriever,batch['history'],candidates,
                              self.config.get('evaluation',{}).get('catalog_chunk',4096),exclude)
        scores = rerank(self.ranker,batch,ids,self.catalog,self.statistics,self.ranking_state['config']['ranking'],
                        self.config.get('evaluation',{}).get('candidate_chunk',64))
        responses = []
        with sqlite3.connect(f'file:{self.root / "events.db"}?mode=ro',uri=True) as db:
            for i,request in enumerate(requests):
                # Per-request candidate limits remain identical whether served alone or in a batch.
                order = scores[i,:request.candidates].argsort(descending=True,stable=True)[:request.k]
                selected = [(int(ids[i,j]),float(scores[i,j])) for j in order if ids[i,j]>=2 and torch.isfinite(scores[i,j])]
                items = [{'item_id':db.execute('SELECT raw FROM item_vocab WHERE id=?',(item,)).fetchone()[0],
                          'score':score} for item,score in selected]
                responses.append({'items':items,'dataset_version':self.version,'models':self.model_versions,
                                  'primary_label':self.manifest['primary_label']})
        return responses

    @torch.inference_mode()
    def predict(self,request):
        if 'watch' in self.ranker.tasks and self.ranker.watch_head.mode in {'normalized','duration_conditioned'} and request.duration<=0:
            raise ValueError('This watch model requires positive request duration')
        batch = self.batch([request])
        with sqlite3.connect(f'file:{self.root / "events.db"}?mode=ro',uri=True) as db:
            for key,raw,table in [('item',request.item_id,'item_vocab'),('category',request.category,'category_vocab')]:
                row = db.execute(f'SELECT id FROM {table} WHERE raw=?',(raw,)).fetchone()
                batch[key] = torch.tensor([row[0] if row else 1],device=self.device)
        batch['duration'] = torch.tensor([request.duration],device=self.device)
        array = dense_features(np.array([request.duration]),np.array([request.timestamp]),
                               *[batch[k].cpu().numpy() for k in ('history','history_watch','history_duration','history_time')])
        batch['dense'] = torch.from_numpy(array).to(self.device)
        prediction = self.ranker(transform_batch(batch,self.statistics))
        return {'predictions':{k:float(v[0]) for k,v in prediction.items() if k in {'binary','conversion','ctcvr','watch'}},
                'primary_label':self.manifest['primary_label'],'watch_condition':self.ranking_state.get('watch_artifact',{}).get('condition') if self.ranking_state.get('watch_artifact') else None,
                'dataset_version':self.version,'models':self.model_versions}


def create_app(config,bundle=None):
    bundle = bundle or Bundle(config)
    app = FastAPI(title='SeqRank research v2')
    app.state.bundle = bundle
    @app.get('/health')
    def health():
        return {'status':'ready','dataset_version':bundle.version,'models':bundle.model_versions}
    @app.post('/recommend')
    def recommend(request: Request):
        try:
            return bundle.recommend([request])[0]
        except ValueError as exc:
            raise HTTPException(422,str(exc)) from exc
    @app.post('/recommend-batch')
    def batch(requests: Annotated[list[Request],Field(min_length=1,max_length=64)]):
        try:
            return bundle.recommend(requests)
        except ValueError as exc:
            raise HTTPException(422,str(exc)) from exc
    @app.post('/predict')
    def predict(request: PredictRequest):
        try:
            return bundle.predict(request)
        except ValueError as exc:
            raise HTTPException(422,str(exc)) from exc
    return app


def benchmark(config):
    """In-process real validation histories; excludes HTTP transport and startup."""
    from .dataset import ResearchDataset
    from .training import loader, move
    bundle = Bundle(config)
    spec = config.get('benchmark',{})
    durations=[]
    rows=0
    for i,batch in enumerate(loader(ResearchDataset(bundle.root,'valid'),config)):
        if i>=spec.get('batches',20)+1:
            break
        batch = move(batch,bundle.device)
        if bundle.device.type=='cuda':
            torch.cuda.synchronize(bundle.device)
        start=time.perf_counter()
        with torch.inference_mode():
            _,ids=exact_topk(bundle.retriever,batch['history'],spec.get('candidates',100))
            rerank(bundle.ranker,batch,ids,bundle.catalog,bundle.statistics,bundle.ranking_state['config']['ranking'])
        if bundle.device.type=='cuda':
            torch.cuda.synchronize(bundle.device)
        if i:
            durations.append(time.perf_counter()-start)
            rows+=len(batch['item'])
    if not durations:
        raise ValueError('Need >=2 validation batches')
    result={'batch_latency_seconds_p50':float(np.quantile(durations,.5)),
            'batch_latency_seconds_p95':float(np.quantile(durations,.95)),
            'requests_per_second':rows/sum(durations),'batches':len(durations),
            'models':bundle.model_versions,'dataset_version':bundle.version,
            'scope':'exact retrieval + ranking; warm-up excluded; no HTTP/feature-store latency'}
    write_json(spec.get('output',str(Path(config['run']['output'])/'benchmark.json')),result)
    return result
