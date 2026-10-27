"""Task conflict diagnostics without mutating parameters or .grad buffers."""
from __future__ import annotations

from pathlib import Path

import torch

from .config import write_json
from .dataset import ResearchDataset, load_artifacts
from .evaluation import checkpoint_path
from .features import transform_batch
from .objectives import components
from .training import load_checkpoint, loader, move


def task_gradients(model,batch,config):
    was_training = model.training
    model.eval()  # deterministic shared features: dropout is disabled for diagnostic only
    try:
        terms = components(model,batch,config,'ranking')
        params = model.shared_parameters()
        vectors={}
        losses={}
        counts={}
        for task,(numerator,count) in terms.items():
            counts[task]=int(count)
            if count==0:
                continue
            loss = numerator/count
            gradients = torch.autograd.grad(loss,params,retain_graph=True,allow_unused=True)
            vectors[task] = torch.cat([(g if g is not None else torch.zeros_like(p)).detach().flatten() for p,g in zip(params,gradients)])
            losses[task]=float(loss.detach())
        norms={task:float(v.norm()) for task,v in vectors.items()}
        cosines={f'{a}:{b}':float(torch.dot(vectors[a],vectors[b])/(vectors[a].norm()*vectors[b].norm()).clamp_min(1e-12))
                 for a in vectors for b in vectors if a<b}
        return {'losses':losses,'valid_counts':counts,'gradient_norms':norms,'gradient_cosines':cosines,
                'scope':'unweighted task means on shared parameters; eval mode; zero norm cosine defined as 0'}
    finally:
        model.train(was_training)


def diagnose(config):
    manifest,statistics,_=load_artifacts(config['data']['prepared'])
    device=config.get('evaluation',{}).get('device','cpu')
    model,state=load_checkpoint(checkpoint_path(config,'ranking'),device,manifest['fingerprint'])
    results=[]
    for i,batch in enumerate(loader(ResearchDataset(config['data']['prepared'],'valid'),config)):
        if i>=config.get('diagnostics',{}).get('batches',10):
            break
        results.append(task_gradients(model,transform_batch(move(batch,device),statistics),state['config']))
    result={'dataset_hash':manifest['fingerprint'],'batches':results}
    write_json(config.get('diagnostics',{}).get('output',str(Path(config['run']['output'])/'gradients.json')),result)
    return result
