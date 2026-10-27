"""Neural retrieval and ranking with explicit task topology and tensor contracts."""
from __future__ import annotations

import math

import torch
from torch import nn

from ..models import SequenceEncoder
from .watchtime import WatchHead


class NeuralRetriever(nn.Module):
    """GRU4Rec-inspired or SASRec-inspired; shared binary sampling objective."""
    def __init__(self, n_items: int, config: dict):
        super().__init__()
        self.kind = config.get('model','sasrec')
        dim = config.get('dim',64)
        if dim<1:
            raise ValueError('Embedding dimension must be positive')
        if self.kind == 'sasrec':
            if dim%config.get('heads',2):
                raise ValueError('SASRec dim must be divisible by heads')
            self.encoder = SequenceEncoder(n_items, **{k:v for k,v in config.items() if k != 'model'})
        elif self.kind == 'gru4rec':
            self.item = nn.Embedding(n_items,dim,padding_idx=0)
            self.gru = nn.GRU(dim,dim,num_layers=config.get('layers',1),batch_first=True)
            self.bos = nn.Parameter(torch.zeros(1,1,dim))
        else:
            raise ValueError(f'Unknown neural retriever: {self.kind}')
        self.dim, self.n_items = dim, n_items

    def item_vectors(self, ids: torch.Tensor | None = None) -> torch.Tensor:
        embedding = self.encoder.item if self.kind == 'sasrec' else self.item
        return embedding.weight if ids is None else embedding(ids)

    def user_vectors(self, history: torch.Tensor) -> torch.Tensor:
        """history [B,L] right padded -> user [B,D]."""
        if self.kind == 'sasrec':
            return self.encoder(history)
        x = torch.cat([self.bos.expand(history.size(0),-1,-1),self.item(history)],1)
        output, _ = self.gru(x)
        return output[torch.arange(history.size(0),device=history.device),(history!=0).sum(-1)]

    def score(self, query: torch.Tensor, ids: torch.Tensor) -> torch.Tensor:
        vectors = self.item_vectors(ids)
        return torch.einsum('bd,bkd->bk',query,vectors)/math.sqrt(self.dim) if ids.ndim==2 else query@vectors.T/math.sqrt(self.dim)

    def forward(self, history: torch.Tensor, candidates: torch.Tensor | None = None) -> torch.Tensor:
        query = self.user_vectors(history)
        return query if candidates is None else self.score(query,candidates)


class RankingNetwork(nn.Module):
    """Shared feature encoder; DNN/DIN, shared-bottom or MMoE task representations.

    batch: user/item/category [B], history [B,L], standardized dense [B,8].
    Disabled tasks have no parameters and never emit fake predictions.
    """
    def __init__(self, manifest: dict, config: dict, watch_artifact: dict | None):
        super().__init__()
        self.spec = config
        if any(not isinstance(w,(int,float)) or w<0 for w in config['tasks'].values()):
            raise ValueError('Task weights must be nonnegative numbers')
        if config.get('experts',4)<1:
            raise ValueError('At least one expert required')
        self.kind = config.get('model','mmoe')
        if self.kind not in {'dnn','din','shared_bottom','mmoe'}:
            raise ValueError(f'Unknown ranking model: {self.kind}')
        self.tasks = [k for k,w in config['tasks'].items() if w>0]
        if not self.tasks or any(t not in {'binary','conversion','watch'} for t in self.tasks):
            raise ValueError('Enable at least one of binary/conversion/watch')
        if 'conversion' in self.tasks and ('binary' not in self.tasks or manifest['primary_label']!='click'):
            raise ValueError('ESMM needs binary(click) and conversion together')
        if 'conversion' in self.tasks and not manifest['train_labels']['conversion']:
            raise ValueError('No observed conversion labels: disable conversion task')
        dim = config.get('dim',64)
        self.user = nn.Embedding(manifest['n_users'],dim,padding_idx=0)
        self.item = nn.Embedding(manifest['n_items'],dim,padding_idx=0)
        self.category = nn.Embedding(manifest['n_categories'],dim,padding_idx=0)
        self.use_attention = config.get('attention',self.kind!='dnn')
        if self.use_attention:
            self.attention = nn.Sequential(nn.Linear(4*dim,dim),nn.PReLU(),nn.Linear(dim,1))
        width = dim*5+8
        hidden = config.get('hidden',128)
        dropout = config.get('dropout',.1)
        def mlp():
            return nn.Sequential(nn.Linear(width,hidden),nn.ReLU(),nn.Dropout(dropout),nn.Linear(hidden,dim),nn.ReLU())
        if self.kind=='mmoe':
            self.experts = nn.ModuleList([mlp() for _ in range(config.get('experts',4))])
            self.gates = nn.ModuleDict({task:nn.Linear(width,len(self.experts)) for task in self.tasks})
        else:
            self.bottom = mlp()
        self.towers = nn.ModuleDict({task:nn.Sequential(nn.Linear(dim,dim),nn.ReLU()) for task in self.tasks})
        self.heads = nn.ModuleDict({task:nn.Linear(dim,1) for task in self.tasks if task!='watch'})
        if 'watch' in self.tasks:
            if watch_artifact is None:
                raise ValueError('Missing train-fitted watch transform')
            self.watch_head = WatchHead(dim,watch_artifact)

    def encode_history(self, history: torch.Tensor) -> torch.Tensor:
        return self.item(history)  # [B,L,D], computed once for candidate reranking

    def forward(self, batch: dict, history_embedding: torch.Tensor | None = None) -> dict:
        h = self.encode_history(batch['history']) if history_embedding is None else history_embedding
        target = self.item(batch['item'])
        mask = batch['history']!=0
        if self.use_attention:
            t = target[:,None].expand_as(h)
            logits = self.attention(torch.cat([h,t,h-t,h*t],-1)).squeeze(-1).float()
            weights = logits.masked_fill(~mask,-1e4).softmax(-1)*mask
        else:
            weights = mask.float()
        weights = weights/weights.sum(-1,keepdim=True).clamp_min(1e-8)
        interest = (weights[...,None]*h).sum(1)
        x = torch.cat([self.user(batch['user']), target, self.category(batch['category']),
                       interest, target*interest, batch['dense']],-1)
        if self.kind=='mmoe':
            experts = torch.stack([m(x) for m in self.experts],1)
            representations = {task:(self.gates[task](x).softmax(-1)[...,None]*experts).sum(1) for task in self.tasks}
        else:
            bottom = self.bottom(x)
            representations = {task:bottom for task in self.tasks}
        result = {}
        for task in self.tasks:
            representation = self.towers[task](representations[task])
            if task=='watch':
                result.update(self.watch_head(representation,batch['duration']))
            else:
                logits = self.heads[task](representation).squeeze(-1).float()
                result[task+'_logit'] = logits
                result[task] = logits.sigmoid()
        if 'conversion' in self.tasks:
            result['ctcvr'] = result['binary']*result['conversion']
        return result

    def shared_parameters(self):
        """Exclude task gates/towers/heads; diagnostics only compare truly shared parameters."""
        modules = [self.user,self.item,self.category]
        if self.use_attention:
            modules.append(self.attention)
        modules += list(self.experts) if self.kind=='mmoe' else [self.bottom]
        return [p for module in modules for p in module.parameters() if p.requires_grad]
