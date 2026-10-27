"""Task sums/counts enable exact global valid-sample means across accumulation/DDP."""
from __future__ import annotations

import torch
from torch.nn import functional as F

from .watchtime import watch_mask


class NegativeSampler:
    def __init__(self, n_items: int, popularity: torch.Tensor, kind='uniform', power=.75):
        if n_items < 4 or kind not in {'uniform','popularity'}:
            raise ValueError('Need >=2 catalog items; sampler must be uniform or popularity')
        weights = torch.ones(n_items-2,dtype=torch.float64) if kind=='uniform' else (popularity[2:].double()+1).pow(power)
        self.probability = weights/weights.sum()
        self.cdf = self.probability.cumsum(0)
        self.cdf[-1] = 1
        self.n_items = n_items

    def sample(self, target: torch.Tensor, count: int) -> torch.Tensor:
        """Conditional inverse CDF: exact target exclusion, no rejection loop or B*I tensor."""
        if count < 1 or ((target < 2)|(target >= self.n_items)).any():
            raise ValueError('Positive candidate must belong to training catalog; count>=1')
        p = self.probability.to(target.device)
        cdf = self.cdf.to(target.device)
        index = target-2
        left = cdf[index]-p[index]
        u = torch.rand(target.size(0),count,device=target.device,dtype=torch.float64)*(1-p[index,None])
        u = u+(u>=left[:,None])*p[index,None]
        return torch.searchsorted(cdf,u.contiguous(),right=True).clamp_max(len(cdf)-1)+2


def masks(batch: dict, config: dict, stage: str) -> dict:
    if stage=='retrieval':
        return {'retrieval':(batch['label']==1)&(batch['item']>=2)}
    result = {}
    for task, weight in config['ranking']['tasks'].items():
        if weight<=0:
            continue
        if task=='binary':
            result[task] = batch['label']>=0
        elif task=='conversion':
            result[task] = (batch['conversion']>=0)&(batch['click']>=0)
        elif task=='watch':
            result[task] = watch_mask(batch,config['watchtime'])
        else:
            raise ValueError(f'Unknown task {task}')
    return result


def components(model, batch: dict, config: dict, stage: str, sampler=None) -> dict:
    raw = model.module if hasattr(model,'module') else model
    valid = masks(batch,config,stage)
    if stage=='retrieval':
        spec = config['retrieval']
        target = batch['item'].clamp_min(2)
        objective = spec.get('loss','sampled_binary')
        if objective=='sampled_binary':
            negatives = sampler.sample(target,int(spec.get('negatives',64)))
            candidates = torch.cat([target[:,None],negatives],1)
            scores = model(batch['history'],candidates).float()
            loss = F.softplus(-scores[:,0])+F.softplus(scores[:,1:]).mean(-1)
        elif objective=='full_softmax':
            candidates = torch.arange(2,raw.n_items,device=target.device)
            scores = model(batch['history'],candidates).float()
            loss = F.cross_entropy(scores,target-2,reduction='none')
        else:
            raise ValueError('loss must be sampled_binary or full_softmax; sampled softmax is not implemented')
        return {'retrieval':((loss*valid['retrieval']).sum(),valid['retrieval'].sum())}
    output = model(batch)
    result = {}
    for task, mask in valid.items():
        if task=='binary':
            values = F.binary_cross_entropy_with_logits(output['binary_logit'],batch['label'].clamp_min(0),reduction='none')
        elif task=='conversion':
            log_joint = (F.logsigmoid(output['binary_logit'])+F.logsigmoid(output['conversion_logit'])).clamp_max(-1e-7)
            y = batch['conversion'].clamp_min(0)
            values = -y*log_joint-(1-y)*torch.log(-torch.expm1(log_joint))
        else:
            values = raw.watch_head.per_sample_loss(output,batch)
        result[task] = ((values*mask).sum(),mask.sum())
    return result


def utility(output: dict, config: dict) -> torch.Tensor:
    """Identical fusion for offline candidates/API; no untrained-head contribution."""
    terms = []
    for task, weight in config.get('fusion',{'binary':1.,'watch':.01}).items():
        if not weight:
            continue
        key = 'ctcvr' if task=='conversion' else task
        if key not in output:
            raise ValueError(f'Fusion references disabled task: {task}')
        value = output[key]
        if task=='watch':
            value = value.log1p()
        terms.append(float(weight)*value)
    if not terms:
        raise ValueError('At least one enabled fusion term required')
    return sum(terms)
