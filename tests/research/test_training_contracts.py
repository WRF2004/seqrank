"""Additional loss/feature contracts; these are authored tests, not executed results."""
from copy import deepcopy

import torch

from seqrank.research.dataset import ResearchDataset, load_artifacts
from seqrank.research.features import candidate_batch, transform_batch
from seqrank.research.networks import RankingNetwork
from seqrank.research.objectives import components, masks
from seqrank.research.training import loader
from seqrank.research.watchtime import fit_watchtime


def test_accumulation_uses_valid_count(prepared):
    manifest,stats,_=load_artifacts(prepared['data']['prepared'])
    data=ResearchDataset(prepared['data']['prepared'],'train')
    model=RankingNetwork(manifest,prepared['ranking'],fit_watchtime(data,prepared['watchtime'])).eval()
    batch=transform_batch(next(iter(loader(data,prepared))),stats)
    batch['label'][::3]=-1
    batch['dwell'][::2]=-1
    whole=components(model,batch,prepared,'ranking')
    full_loss=sum(n/c.clamp_min(1) for n,c in whole.values())
    params=list(model.parameters())
    full_grad=torch.autograd.grad(full_loss,params,allow_unused=True)
    accumulated=[torch.zeros_like(p) for p in params]
    for start,stop in [(0,3),(3,len(batch['item']))]:
        micro={k:v[start:stop] for k,v in batch.items()}
        partial=components(model,micro,prepared,'ranking')
        loss=sum(n/whole[t][1].clamp_min(1) for t,(n,_) in partial.items())
        grads=torch.autograd.grad(loss,params,allow_unused=True)
        for target,g in zip(accumulated,grads):
            if g is not None:
                target.add_(g)
    for expected,actual in zip(full_grad,accumulated):
        if expected is not None:
            torch.testing.assert_close(expected,actual,rtol=1e-4,atol=1e-6)


def test_task_disable_and_empty_masks(prepared):
    config=deepcopy(prepared)
    config['ranking']['tasks']={'binary':1,'conversion':0,'watch':0}
    manifest,stats,_=load_artifacts(config['data']['prepared'])
    model=RankingNetwork(manifest,config['ranking'],None)
    data=ResearchDataset(config['data']['prepared'],'train')
    batch=transform_batch(next(iter(loader(data,config))),stats)
    assert set(model(batch))=={'binary','binary_logit'}
    assert not hasattr(model,'watch_head') and 'conversion' not in model.heads
    batch['label'].fill_(-1)
    n,c=components(model,batch,config,'ranking')['binary']
    assert c==0 and n==0
    n.backward()
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)


def test_candidate_feature_parity(prepared):
    manifest,statistics,catalog=load_artifacts(prepared['data']['prepared'])
    data=ResearchDataset(prepared['data']['prepared'],'valid')
    batch=next(iter(loader(data,prepared)))
    ids=batch['item'][:,None]
    candidates=candidate_batch(batch,ids,catalog,statistics)
    # Use catalog feature snapshots on both sides: candidates have no current exposure label.
    from seqrank.research.features import dense_features
    expected=dict(batch)
    expected['duration']=candidates['duration']
    expected['category']=candidates['category']
    expected['dense']=torch.from_numpy(dense_features(expected['duration'].numpy(),expected['timestamp'].numpy(),
        *[expected[k].numpy() for k in ['history','history_watch','history_duration','history_time']]))
    expected=transform_batch(expected,statistics)
    for field in ['item','history','user','category','dense','duration']:
        torch.testing.assert_close(candidates[field],expected[field])
    assert 'dwell' not in candidates and 'label' not in candidates


def test_watch_fitting_not_changed_by_test_labels(prepared):
    data=ResearchDataset(prepared['data']['prepared'],'train')
    before=fit_watchtime(data,prepared['watchtime'])
    # A different held-out tensor is deliberately not an argument to the fitter.
    test=ResearchDataset(prepared['data']['prepared'],'test')
    heldout=test[0]
    heldout['dwell'].fill_(999999)
    after=fit_watchtime(data,prepared['watchtime'])
    assert before==after


def test_observed_watch_vs_conditional(prepared):
    batch=next(iter(loader(ResearchDataset(prepared['data']['prepared'],'train'),prepared)))
    batch['click'].fill_(0)
    config=deepcopy(prepared)
    config['watchtime']['condition']='click'
    assert masks(batch,config,'ranking')['watch'].sum()==0
    config['watchtime']['condition']='observed'
    assert masks(batch,config,'ranking')['watch'].sum()>0
