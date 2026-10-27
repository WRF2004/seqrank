"""Observed-exposure prediction and full-catalogue retrieval are separate protocols."""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import torch

from .config import empty_output, file_hash, write_json, fingerprint
from .dataset import ResearchDataset, load_artifacts
from .features import candidate_batch, transform_batch
from .metrics import MetricStore, RetrievalMetrics
from .objectives import utility
from .retrievers import ANN, exact_topk
from .training import load_checkpoint, loader, move
from .watchtime import watch_mask


def checkpoint_path(config,stage):
    return config.get('checkpoints',{}).get(stage,str(Path(config['run']['output'])/stage/'best.pt'))


@torch.inference_mode()
def rerank(model,batch,items,catalog,statistics,spec,chunk=128):
    """Return scores [B,C]. Reuse target-independent history embeddings across candidates."""
    if chunk<1:
        raise ValueError('candidate chunk must be positive')
    h = model.encode_history(batch['history'])
    scores = []
    for start in range(0,items.size(1),chunk):
        ids = items[:,start:start+chunk]
        features = candidate_batch(batch,ids,catalog,statistics)
        embedded = h[:,None].expand(-1,ids.size(1),-1,-1).reshape(-1,h.size(1),h.size(2))
        value = utility(model(features,embedded),spec).reshape_as(ids)
        invalid = ids<2
        if spec.get('fusion',{}).get('watch',0) and model.watch_head.mode in {'normalized','duration_conditioned'}:
            invalid |= features['duration'].reshape_as(ids)<=0
        scores.append(value.masked_fill(invalid,-torch.inf))
    return torch.cat(scores,1)


@torch.inference_mode()
def evaluate(config,stage,split='valid'):
    if stage not in {'retrieval','ranking','e2e'}:
        raise ValueError('Unknown evaluation stage')
    if split not in {'valid','test'}:
        raise ValueError('Evaluation split must be valid or test')
    spec = config.get('evaluation',{})
    device = torch.device(spec.get('device','cpu'))
    manifest,statistics,catalog = load_artifacts(config['data']['prepared'])
    dataset = ResearchDataset(config['data']['prepared'],split)
    path = checkpoint_path(config,'ranking' if stage=='ranking' else 'retrieval')
    model,state = load_checkpoint(path,device,manifest['fingerprint'])
    from .contracts import check_configuration, require_final
    check_configuration(config,state,manifest)
    if split=='test':
        require_final(config,manifest,state['stage'],path)
    output = empty_output(str(spec.get('output',str(Path(config['run']['output'])/'evaluation')))+f'/{stage}-{split}')
    expected_stage = 'ranking' if stage=='ranking' else 'retrieval'
    if state['stage']!=expected_stage:
        raise ValueError('Incorrect checkpoint stage')
    result = {'stage':stage,'split':split,'dataset_hash':manifest['fingerprint'],
              'checkpoint_hash':file_hash(path),'config':config,'primary_label':manifest['primary_label'],
              'code_commit':state['code_commit'],'code_diff_hash':state.get('code_diff_hash'),'seed':state['config'].get('seed',42)}
    from .contracts import evaluation_protocol
    result['evaluation_protocol'] = evaluation_protocol(config)
    result['evaluation_protocol_fingerprint'] = fingerprint(result['evaluation_protocol'])
    started = time.perf_counter()
    if stage=='ranking':
        store = MetricStore(output/'predictions.sqlite')
        watch = state.get('watch_artifact')
        tail_count = watch_count = 0
        try:
            for raw in loader(dataset,config):
                batch = transform_batch(move(raw,device),statistics)
                predictions = model(batch)
                users = raw['group_user'].numpy()
                if 'binary' in predictions:
                    store.binary(manifest['primary_label'],users,raw['label'].numpy(),predictions['binary'].cpu().numpy())
                if 'conversion' in predictions:
                    cv = raw['conversion'].numpy().copy()
                    cv[raw['click'].numpy()!=1] = -1
                    store.binary('conditional_cvr',users,cv,predictions['conversion'].cpu().numpy())
                    store.binary('ctcvr',users,raw['conversion'].numpy(),predictions['ctcvr'].cpu().numpy())
                if watch:
                    mask = watch_mask(batch,watch['spec']).cpu().numpy()
                    y = raw['dwell'].numpy()[mask]
                    p = predictions['watch'].cpu().numpy()[mask]
                    bucket = np.searchsorted(watch['edges'][1:-1],y,side='right')
                    store.watch(users[mask],y,p,raw['duration'].numpy()[mask],
                                (raw['history'].numpy()!=0).sum(-1)[mask],raw['dense'].numpy()[mask,3],bucket)
                    tail_count += int((y>watch['cap_seconds']).sum())
                    watch_count += len(y)
            result['metrics'] = store.result(watch['duration_edges'] if watch else [],watch['edges'][1:-1] if watch else [])
            result['watch_labels_above_training_cap'] = tail_count
            result['watch_evaluation_count'] = watch_count
            result['protocol'] = 'observed exposures; watch error uses uncapped observed seconds; no unexposed negatives'
        finally:
            store.close()
    else:
        ks = spec.get('ks',[10,50,100])
        metrics = {key:RetrievalMetrics(ks,manifest['n_items']) for key in ('all','cold_user','warm_user','long_tail','head')}
        popularity = np.load(Path(config['data']['prepared'])/'popularity.npy')
        threshold = float(np.quantile(popularity[2:],spec.get('tail_quantile',.8)))
        ranking = None
        if stage=='e2e':
            rankpath = checkpoint_path(config,'ranking')
            ranking,rankstate = load_checkpoint(rankpath,device,manifest['fingerprint'])
            check_configuration(config,rankstate,manifest)
            if split=='test':
                require_final(config,manifest,'ranking',rankpath)
            if rankstate['stage']!='ranking':
                raise ValueError('Expected ranking checkpoint')
            result['ranking_checkpoint_hash'] = file_hash(rankpath)
        target_count=oov_count=0
        for raw in loader(dataset,config):
            mask = raw['label']==1
            if not mask.any():
                continue
            batch = move({k:v[mask] for k,v in raw.items()},device)
            k = max(max(ks),int(spec.get('candidates',100))) if ranking else max(ks)
            _,ids = exact_topk(model,batch['history'],k,int(spec.get('catalog_chunk',4096)),spec.get('exclude_seen',False))
            if ranking:
                scores = rerank(ranking,batch,ids,catalog,statistics,rankstate['config']['ranking'],int(spec.get('candidate_chunk',64)))
                ids = ids.masked_fill(~torch.isfinite(scores),0)
                ids = ids.gather(1,scores.argsort(dim=1,descending=True,stable=True))
            targets = batch['item'].cpu().numpy()
            cold = batch['user'].cpu().numpy()==1
            tail = (targets>=2)&(popularity[targets]<=threshold)
            ids = ids.cpu().numpy()
            groups = {'all':np.ones(len(targets),bool),'cold_user':cold,'warm_user':~cold,'long_tail':tail,'head':(targets>=2)&~tail}
            for key,m in groups.items():
                metrics[key].update(targets[m],ids[m])
            target_count+=len(targets)
            oov_count+=int((targets<2).sum())
        result['metrics'] = {key:m.result() for key,m in metrics.items()}
        result.update(targets=target_count,oov_targets=oov_count,long_tail_train_count_threshold=threshold,
                      protocol='fixed retrieval candidates, one observed positive; reranking proxy only, no online uplift claim' if ranking else 'exact full training catalogue, one observed positive, OOV misses')
    result['elapsed_seconds'] = time.perf_counter()-started
    write_json(output/'metrics.json',result)
    return result


@torch.inference_mode()
def benchmark_ann(config):
    spec = config['ann']
    evaluation = config.get('evaluation',{})
    device = torch.device(evaluation.get('device','cpu'))
    manifest,_,_ = load_artifacts(config['data']['prepared'])
    path = checkpoint_path(config,'retrieval')
    model,_ = load_checkpoint(path,device,manifest['fingerprint'])
    ann = ANN(spec['path'],file_hash(path),manifest['fingerprint'],spec.get('ef_search',128))
    exact_times=[]
    ann_times=[]
    overlaps=[]
    k = int(spec.get('k',100))
    def sync():
        if device.type=='cuda':
            torch.cuda.synchronize(device)
    for index,batch in enumerate(loader(ResearchDataset(config['data']['prepared'],'valid'),config)):
        if index>=int(spec.get('benchmark_batches',20))+1:
            break
        history = batch['history'].to(device)
        sync()
        start=time.perf_counter()
        _,exact = exact_topk(model,history,k)
        sync()
        exact_elapsed=time.perf_counter()-start
        start=time.perf_counter()
        query = model.user_vectors(history).cpu().numpy()
        _,approx = ann.search(query,k)
        sync()
        ann_elapsed=time.perf_counter()-start
        if index==0:
            continue  # warm-up excluded
        exact_times.append(exact_elapsed)
        ann_times.append(ann_elapsed)
        overlaps.extend(len(set(a)&set(b))/max(1,len(a)) for a,b in zip(exact.cpu().numpy(),approx))
    if not overlaps:
        raise ValueError('Need at least two validation batches for warm-up and benchmark')
    result = {'candidate_overlap_mean':float(np.mean(overlaps)), 'queries':len(overlaps),
              'exact_batch_seconds':exact_times,'ann_batch_seconds_including_query_and_transfer':ann_times,
              'index':ann.meta,'memory_definition':'serialized index bytes; not peak resident memory',
              'dataset_hash':manifest['fingerprint'],'checkpoint_hash':file_hash(path)}
    write_json(spec.get('benchmark_output',str(Path(config['run']['output'])/'ann-benchmark.json')),result)
    return result
