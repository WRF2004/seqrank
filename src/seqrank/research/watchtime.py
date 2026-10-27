"""Watch-time targets, training-only discretization and executable restoration.

cread_inspired follows threshold classification + restoration + ordinal penalty;
empirical CDF/reservoir discretization and normalized Huber are stated deviations.
"""
from __future__ import annotations

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

MODES = {'raw','log1p','normalized','bucket','cread_inspired','duration_conditioned'}


def watch_mask(batch: dict, spec: dict) -> torch.Tensor:
    mask = torch.isfinite(batch['dwell']) & (batch['dwell'] >= 0)
    condition = spec.get('condition', 'click')
    if condition == 'click':
        mask &= batch['click'] == 1
    elif condition == 'positive':
        mask &= batch['label'] == 1
    elif condition != 'observed':
        raise ValueError('watchtime.condition must be click, positive or observed')
    if spec.get('mode') in {'normalized','duration_conditioned'}:
        mask &= batch['duration'] > 0
    return mask


def adaptive_edges(y: np.ndarray, bins: int, beta: float, cap: float) -> tuple[np.ndarray, float]:
    """Empirical EAD objective over calibrated quantiles; normalized seconds domain."""
    if bins < 2 or cap <= 0:
        raise ValueError('Need >=2 bins and positive cap')
    z = np.clip(np.asarray(y, dtype=float)/cap, 0, 1)
    best = None
    for alpha in np.linspace(0, 5, 21):
        q = np.linspace(0, 1, bins+1)
        q = q if alpha == 0 else -np.expm1(-alpha*q)/-np.expm1(-alpha)
        edges = np.unique(np.r_[0, np.quantile(z, q)[1:-1], 1])
        widths = np.diff(edges)
        p = np.histogram(z, edges)[0].astype(float)
        p = (p+.5)/(len(z)+.5*len(p))  # explicit finite-sample smoothing
        objective = float((p*p).sum()*((widths*widths/p).sum()+beta*(widths*widths).sum()))
        if best is None or objective < best[0]:
            best = (objective, edges*cap, float(alpha))
    return best[1], best[2]


def fit_watchtime(dataset, spec: dict, seed: int = 42) -> dict:
    """Bounded reservoir over eligible TRAIN labels; never reads valid/test."""
    if any(not s['file'].startswith('train-') for s in dataset.shards):
        raise ValueError('Watch transform can only fit train shards')
    rng = np.random.default_rng(seed)
    limit = int(spec.get('fit_samples', 200000))
    if limit < 1:
        raise ValueError('fit_samples must be positive')
    samples, count = [], 0
    for i in range(len(dataset)):
        row = dataset[i]
        if not bool(watch_mask({k:v.reshape(1) for k,v in row.items() if k in {'dwell','duration','click','label'}}, spec)[0]):
            continue
        count += 1
        pair = (float(row['dwell']), float(row['duration']))
        if len(samples) < limit:
            samples.append(pair)
        else:
            j = int(rng.integers(count))
            if j < limit:
                samples[j] = pair
    if not samples:
        raise ValueError('No eligible training watch labels for configured condition/mode')
    y, duration = np.array(samples).T
    cap = float(spec.get('cap_seconds', 3600))
    if cap <= 0:
        raise ValueError('cap_seconds must be positive')
    clipped = np.clip(y, 0, cap)
    bins = int(spec.get('bins', 32))
    if bins < 2:
        raise ValueError('watch bins must be >=2')
    mode = spec.get('mode','log1p')
    if mode not in MODES:
        raise ValueError(f'Unknown watchtime mode: {mode}')
    if mode == 'cread_inspired':
        edges, alpha = adaptive_edges(clipped, bins, float(spec.get('ead_beta', 100)), cap)
    else:
        edges = np.unique(np.r_[0, np.quantile(clipped, np.linspace(0,1,bins+1))[1:-1], cap])
        alpha = None
    if len(edges) < 2:
        edges = np.array([0, cap])
    assignment = np.searchsorted(edges[1:-1], clipped, side='right')
    centers = [float(clipped[assignment == i].mean()) if np.any(assignment == i)
               else float((edges[i]+edges[i+1])/2) for i in range(len(edges)-1)]
    valid_d = duration > 0
    duration_edges = np.unique(np.quantile(duration[valid_d], [.25,.5,.75])).tolist() if valid_d.any() else []
    groups = np.searchsorted(duration_edges, duration, side='right')
    logy = np.log1p(clipped)
    mu, sigma = [], []
    for group in range(len(duration_edges)+1):
        values = logy[(groups == group) & valid_d]
        mu.append(float(values.mean()) if len(values) else float(logy.mean()))
        sigma.append(max(float(values.std()) if len(values) else float(logy.std()), .1))
    return {'mode':mode,'condition':spec.get('condition','click'),'edges':edges.tolist(),
            'centers':centers,'alpha':alpha,'cap_seconds':cap,'fit_split':'train',
            'eligible_training_rows':count,'reservoir_size':len(samples),'seed':seed,
            'training_clipped_sample_fraction':float((y>cap).mean()),
            'duration_edges':duration_edges,'duration_mu':mu,'duration_sigma':sigma,
            'spec':spec}


class WatchHead(nn.Module):
    def __init__(self, width: int, artifact: dict):
        super().__init__()
        self.artifact = artifact
        self.mode = artifact['mode']
        self.cap = artifact['cap_seconds']
        self.register_buffer('edges', torch.tensor(artifact['edges'], dtype=torch.float32))
        self.register_buffer('centers', torch.tensor(artifact['centers'], dtype=torch.float32))
        self.register_buffer('duration_edges', torch.tensor(artifact['duration_edges'], dtype=torch.float32))
        self.register_buffer('mu', torch.tensor(artifact['duration_mu'], dtype=torch.float32))
        self.register_buffer('sigma', torch.tensor(artifact['duration_sigma'], dtype=torch.float32))
        n = len(artifact['centers']) if self.mode in {'bucket','cread_inspired'} else 1
        self.project = nn.Linear(width, n)

    def forward(self, representation: torch.Tensor, duration: torch.Tensor) -> dict:
        logits = self.project(representation).float()  # [B,K] or [B,1]
        if self.mode == 'bucket':
            seconds = (logits.softmax(-1)*self.centers).sum(-1)
        elif self.mode == 'cread_inspired':
            # Right Riemann sum of predicted survival, same restoration train and inference.
            seconds = (logits.sigmoid()*self.edges.diff()).sum(-1)
        elif self.mode == 'raw':
            seconds = F.softplus(logits[:,0])
        elif self.mode == 'log1p':
            seconds = torch.expm1(F.softplus(logits[:,0]).clamp_max(np.log1p(self.cap)))
        elif self.mode == 'normalized':
            seconds = F.softplus(logits[:,0])*duration.clamp_min(0)
        else:
            groups = torch.bucketize(duration.contiguous(), self.duration_edges, right=True)
            log_seconds = logits[:,0]*self.sigma[groups]+self.mu[groups]
            seconds = torch.expm1(log_seconds.clamp(0, np.log1p(self.cap)))
        return {'watch_logits':logits, 'watch':seconds.clamp(0,self.cap)}

    def per_sample_loss(self, output: dict, batch: dict) -> torch.Tensor:
        spec = self.artifact['spec']
        y = batch['dwell'].clamp(0,self.cap).float()
        logits = output['watch_logits'].float()
        if self.mode == 'raw':
            loss = F.smooth_l1_loss(F.softplus(logits[:,0]), y, reduction='none')
        elif self.mode == 'log1p':
            loss = F.smooth_l1_loss(F.softplus(logits[:,0]), y.log1p(), reduction='none')
        elif self.mode == 'normalized':
            loss = F.smooth_l1_loss(F.softplus(logits[:,0]), y/batch['duration'].clamp_min(1e-3), reduction='none')
        elif self.mode == 'duration_conditioned':
            groups = torch.bucketize(batch['duration'].contiguous(), self.duration_edges, right=True)
            target = (y.log1p()-self.mu[groups])/self.sigma[groups]
            loss = F.smooth_l1_loss(logits[:,0], target, reduction='none')
        elif self.mode == 'bucket':
            labels = torch.bucketize(y.contiguous(), self.edges[1:-1], right=True)
            loss = F.cross_entropy(logits, labels, reduction='none')
        else:
            labels = (y[:,None] > self.edges[1:]).float()
            ce = F.binary_cross_entropy_with_logits(logits, labels, reduction='none').mean(-1)
            restore = F.smooth_l1_loss(output['watch']/self.cap, y/self.cap, reduction='none')
            p = logits.sigmoid()
            ordinal = F.relu(p[:,1:]-p[:,:-1]).mean(-1) if p.size(1)>1 else logits[:,0]*0
            loss = ce + float(spec.get('restore_weight',1))*restore + float(spec.get('ordinal_weight',.1))*ordinal
        return loss
