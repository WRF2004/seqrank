"""CPU/single GPU/DDP trainer. Global task denominators, AMP and epoch-boundary resume."""
from __future__ import annotations

import contextlib
import hashlib
import json
import math
import os
import random
import subprocess
import time
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler

from ..engine import EvalSampler
from .config import fingerprint, write_json
from .dataset import ResearchDataset, load_artifacts
from .features import transform_batch
from .networks import NeuralRetriever, RankingNetwork
from .objectives import NegativeSampler, components, masks
from .retrievers import CatalogueBaseline, fit_baseline
from .watchtime import fit_watchtime


def move(batch,device):
    return {k:v.to(device,non_blocking=True) for k,v in batch.items()}


def weights(config,stage):
    return {'retrieval':1.} if stage=='retrieval' else {k:float(v) for k,v in config['ranking']['tasks'].items() if v>0}


def capture_rng(device):
    state = np.random.get_state()
    return {'python':random.getstate(),'numpy':[state[0],state[1].tolist(),state[2],state[3],state[4]],
            'torch':torch.get_rng_state(),'cuda':torch.cuda.get_rng_state(device) if device.type=='cuda' else None}


def restore_rng(state,device):
    random.setstate(state['python'])
    n = state['numpy']
    np.random.set_state((n[0],np.array(n[1],dtype=np.uint32),n[2],n[3],n[4]))
    torch.set_rng_state(state['torch'].cpu())
    if state['cuda'] is not None:
        torch.cuda.set_rng_state(state['cuda'].cpu(),device)


def configuration_identity(config):
    # Output paths and evaluation settings do not affect training. Model/data/trainer do.
    return fingerprint({k:config.get(k) for k in ('training','retrieval','ranking','watchtime','seed')})


def make_model(stage,manifest,config,watch=None):
    if stage=='retrieval':
        return NeuralRetriever(manifest['n_items'],dict(config['retrieval'],max_len=manifest['max_len']))
    return RankingNetwork(manifest,config['ranking'],watch)


def load_checkpoint(path,device='cpu',dataset_hash=None):
    state = torch.load(path,map_location='cpu',weights_only=True)
    if state.get('format_version')!=2:
        raise ValueError('Use legacy load_model for v1 checkpoints')
    if dataset_hash and state['manifest']['fingerprint']!=dataset_hash:
        raise ValueError('Checkpoint/dataset fingerprint mismatch')
    if state.get('baseline'):
        return CatalogueBaseline(state['baseline']),state
    model = make_model(state['stage'],state['manifest'],state['config'],state.get('watch_artifact')).to(device)
    model.load_state_dict(state['model'])
    return model.eval(),state


def loader(dataset,config,rank=0,world=1,training=False,epoch=0):
    sampler = DistributedSampler(dataset,world,rank,shuffle=True,seed=config.get('seed',42)) if training else EvalSampler(len(dataset),rank,world)
    if training:
        sampler.set_epoch(epoch)
    workers = config['training'].get('workers',0) if training else 0
    generator = torch.Generator().manual_seed(config.get('seed',42)+epoch*world+rank)
    return DataLoader(dataset,batch_size=config['training']['batch_size'],sampler=sampler,
                      num_workers=workers,pin_memory=torch.cuda.is_available(),generator=generator,
                      multiprocessing_context='spawn' if workers else None)


def atomic_checkpoint(path,state):
    path = Path(path)
    temporary = path.with_suffix('.tmp')
    torch.save(state,temporary)
    temporary.replace(path)


def train(config: dict, stage: str, resume: str | None = None, stop_after: int | None = None):
    if stage not in {'retrieval','ranking'}:
        raise ValueError('stage must be retrieval or ranking')
    spec = config['training']
    deterministic = bool(spec.get('deterministic',True))
    if deterministic:
        os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
    torch.use_deterministic_algorithms(deterministic)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = deterministic
    world,rank,local = [int(os.getenv(k,'1' if k=='WORLD_SIZE' else '0')) for k in ('WORLD_SIZE','RANK','LOCAL_RANK')]
    requested = spec.get('device','cpu')
    device = torch.device(f'cuda:{local}' if world>1 and requested.startswith('cuda') else requested)
    if device.type=='cuda':
        torch.cuda.set_device(device)
    if world>1:
        dist.init_process_group('nccl' if device.type=='cuda' else 'gloo')
    try:
        _train(config,stage,resume,stop_after,device,world,rank)
    finally:
        if world>1 and dist.is_initialized():
            dist.destroy_process_group()


def _train(config,stage,resume,stop_after,device,world,rank):
    spec = config['training']
    seed = int(config.get('seed',42))
    random.seed(seed+rank)
    np.random.seed(seed+rank)
    torch.manual_seed(seed+rank)
    root = config['data']['prepared']
    manifest,statistics,_ = load_artifacts(root)
    dataset,valid = ResearchDataset(root,'train'),ResearchDataset(root,'valid')
    if len(dataset)==0 or len(valid)==0:
        raise ValueError('Nonempty train/valid splits required')
    output = Path(config['run']['output'])/stage
    if output.exists() and any(output.iterdir()) and not resume:
        raise FileExistsError(f'Existing run: {output}. Use a new run.output or explicit --resume')
    output.mkdir(parents=True,exist_ok=True)
    try:
        code = subprocess.check_output(['git','rev-parse','HEAD'],text=True,stderr=subprocess.DEVNULL).strip()
        difference = subprocess.check_output(['git','diff','--binary','HEAD'],stderr=subprocess.DEVNULL)
        code_diff_hash = hashlib.sha256(difference).hexdigest() if difference else None
    except (OSError,subprocess.CalledProcessError):
        code = 'unavailable'
        code_diff_hash = None
    watch = None
    if stage=='ranking':
        tasks = weights(config,stage)
        if 'binary' in tasks and not manifest['train_labels']['label']:
            raise ValueError('No training primary labels')
        if 'watch' in tasks:
            shared = [fit_watchtime(dataset,config['watchtime'],seed) if rank==0 else None]
            if world>1:
                dist.broadcast_object_list(shared,src=0)
            watch = shared[0]
    state_base = {'format_version':2,'stage':stage,'config':config,'manifest':manifest,
                  'statistics':statistics,'watch_artifact':watch,'code_commit':code,'code_diff_hash':code_diff_hash,
                  'configuration_identity':configuration_identity(config),'world_size':world,
                  'torch_version':str(torch.__version__)}
    if stage=='retrieval' and np.load(Path(root)/'popularity.npy').sum()==0:
        raise ValueError('No mature positive retrieval training labels')
    if stage=='retrieval' and config['retrieval']['model'] in {'popularity','itemknn'}:
        if resume or world>1:
            raise ValueError('Count-based baseline fitting is single process; no optimizer resume')
        baseline = fit_baseline(dataset,config['retrieval'])
        atomic_checkpoint(output/'best.pt',dict(state_base,baseline=baseline))
        write_json(output/'config.json',config)
        return
    precision = spec.get('precision','fp32')
    if precision not in {'fp32','fp16','bf16'}:
        raise ValueError('precision must be fp32/fp16/bf16')
    if precision=='fp16' and device.type!='cuda':
        raise ValueError('FP16 training requires CUDA')
    if precision=='bf16' and device.type=='cuda' and not torch.cuda.is_bf16_supported():
        raise ValueError('Selected GPU does not support BF16')
    amp_dtype = torch.bfloat16 if precision=='bf16' else torch.float16
    amp_enabled = precision!='fp32'
    accumulation = int(spec.get('accumulation',1))
    if accumulation<1 or int(spec['epochs'])<1:
        raise ValueError('epochs and accumulation must be positive')
    model = make_model(stage,manifest,config,watch).to(device)
    optimizer = torch.optim.AdamW(model.parameters(),lr=float(spec['lr']),weight_decay=float(spec.get('weight_decay',1e-4)))
    steps_per_epoch = math.ceil(len(loader(dataset,config,rank,world,True))/accumulation)
    total_steps = max(1,steps_per_epoch*int(spec['epochs']))
    warmup = int(spec.get('warmup_steps',0))
    def factor(step):
        if step<warmup:
            return (step+1)/max(1,warmup)
        progress = min(1.,max(0.,(step-warmup)/max(1,total_steps-warmup)))
        return .5*(1+math.cos(math.pi*progress))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer,factor)
    scaler = torch.amp.GradScaler('cuda',enabled=precision=='fp16')
    start,best,step = 0,float('inf'),0
    if resume:
        saved = torch.load(resume,map_location='cpu',weights_only=True)
        if saved['stage']!=stage or saved['manifest']['fingerprint']!=manifest['fingerprint'] or saved['configuration_identity']!=configuration_identity(config) or saved['world_size']!=world:
            raise ValueError('Resume requires identical model/trainer/data/world size configuration')
        model.load_state_dict(saved['model'])
        optimizer.load_state_dict(saved['optimizer'])
        scheduler.load_state_dict(saved['scheduler'])
        scaler.load_state_dict(saved['scaler'])
        start,best,step = saved['epoch']+1,saved['best'],saved['step']
        restore_rng(saved['rng'][rank],device)
    raw = model
    if world>1:
        model = DistributedDataParallel(raw,device_ids=[device.index] if device.type=='cuda' else None)
    sampler = NegativeSampler(manifest['n_items'],torch.from_numpy(np.load(Path(root)/'popularity.npy')),
                              config['retrieval'].get('sampler','uniform'),config['retrieval'].get('popularity_power',.75)) if stage=='retrieval' else None
    task_weights = weights(config,stage)
    if rank==0:
        write_json(output/'config.json',config)
        if watch:
            write_json(output/'watch_transform.json',watch)
    for epoch in range(start,int(spec['epochs'])):
        model.train()
        started = time.perf_counter()
        if device.type=='cuda':
            torch.cuda.reset_peak_memory_stats(device)
        train_loader = loader(dataset,config,rank,world,True,epoch)
        iterator = iter(train_loader)
        totals = {k:torch.zeros(2,device=device,dtype=torch.float64) for k in task_weights}
        nrows = 0
        while True:
            window = []
            for _ in range(accumulation):
                try:
                    window.append(next(iterator))
                except StopIteration:
                    break
            if not window:
                break
            denominators = {task:torch.zeros((),device=device,dtype=torch.float64) for task in task_weights}
            for batch in window:
                for task,mask in masks(batch,config,stage).items():
                    denominators[task] += mask.sum().to(device)
            if world>1:
                for count in denominators.values():
                    dist.all_reduce(count)
            optimizer.zero_grad(set_to_none=True)
            for j,cpu_batch in enumerate(window):
                batch = transform_batch(move(cpu_batch,device),statistics)
                sync = model.no_sync() if world>1 and j<len(window)-1 else contextlib.nullcontext()
                with sync:
                    with torch.autocast(device_type=device.type,dtype=amp_dtype,enabled=amp_enabled):
                        terms = components(model,batch,config,stage,sampler)
                        loss = sum(task_weights[t]*numerator*world/denominators[t].clamp_min(1) for t,(numerator,_) in terms.items())
                    finite = torch.isfinite(loss).to(torch.int32)
                    if world>1:
                        dist.all_reduce(finite,op=dist.ReduceOp.MIN)
                    if not finite.item():
                        raise FloatingPointError('Non-finite loss; all ranks stopped before backward')
                    scaler.scale(loss).backward()
                for task,(numerator,count) in terms.items():
                    totals[task] += torch.stack([numerator.detach().double(),count.double()])
                nrows += len(batch['item'])
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(raw.parameters(),float(spec.get('grad_clip',5)),error_if_nonfinite=precision!='fp16')
            old_scale = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            if scaler.get_scale()>=old_scale:
                scheduler.step()
                step += 1
        raw.eval()
        validation = {k:torch.zeros(2,device=device,dtype=torch.float64) for k in task_weights}
        with torch.random.fork_rng(devices=[device.index] if device.type=='cuda' else []):
            torch.manual_seed(seed+10000+rank)
            with torch.inference_mode():
                for batch in loader(valid,config,rank,world):
                    batch = transform_batch(move(batch,device),statistics)
                    for task,(numerator,count) in components(raw,batch,config,stage,sampler).items():
                        validation[task] += torch.stack([numerator.double(),count.double()])
        if world>1:
            for collection in (totals,validation):
                for pair in collection.values():
                    dist.all_reduce(pair)
        if any(pair[1].item()==0 for pair in validation.values()):
            raise ValueError('An enabled task has zero eligible validation labels; choose valid split/task configuration')
        val_loss = sum(task_weights[t]*(pair[0]/pair[1]).item() for t,pair in validation.items())
        improved = val_loss<best
        best = min(best,val_loss)
        local_rng = capture_rng(device)
        all_rng = [None]*world
        if world>1:
            dist.all_gather_object(all_rng,local_rng)
        else:
            all_rng[0] = local_rng
        elapsed = time.perf_counter()-started
        if rank==0:
            record = {'epoch':epoch,'step':step,'train':{t:(p[0]/p[1].clamp_min(1)).item() for t,p in totals.items()},
                      'valid':{t:(p[0]/p[1]).item() for t,p in validation.items()},'selection_loss':val_loss,
                      'epoch_seconds':elapsed,'rows_per_second_including_sampler_padding':nrows*world/elapsed,
                      'rank0_peak_allocated_bytes':torch.cuda.max_memory_allocated(device) if device.type=='cuda' else None,
                      'lr':scheduler.get_last_lr()[0],'world_size':world,'precision':precision}
            with (output/'metrics.jsonl').open('a') as stream:
                stream.write(json.dumps(record,allow_nan=False)+'\n')
            print(json.dumps(record),flush=True)
            saved = dict(state_base,model=raw.state_dict(),optimizer=optimizer.state_dict(),
                         scheduler=scheduler.state_dict(),scaler=scaler.state_dict(),rng=all_rng,
                         epoch=epoch,best=best,step=step)
            atomic_checkpoint(output/'last.pt',saved)
            if improved:
                atomic_checkpoint(output/'best.pt',saved)
        if world>1:
            dist.barrier()
        if stop_after is not None and epoch+1>=stop_after:
            break
