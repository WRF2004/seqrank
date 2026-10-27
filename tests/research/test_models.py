from copy import deepcopy

import numpy as np
import pytest
import torch

from seqrank.models import SequenceEncoder
from seqrank.research.dataset import ResearchDataset, load_artifacts
from seqrank.research.diagnostics import task_gradients
from seqrank.research.features import transform_batch
from seqrank.research.networks import NeuralRetriever, RankingNetwork
from seqrank.research.objectives import NegativeSampler, components, masks
from seqrank.research.retrievers import CatalogueBaseline, exact_topk, fit_baseline
from seqrank.research.training import loader
from seqrank.research.watchtime import WatchHead, adaptive_edges, fit_watchtime


@pytest.mark.parametrize('kind',['uniform','popularity'])
def test_negative_sampling(kind):
    sampler=NegativeSampler(8,torch.arange(8),kind)
    positives=torch.tensor([2,3,7])
    sample=sampler.sample(positives,1000)
    assert ((sample>=2)&(sample<8)).all()
    assert not (sample==positives[:,None]).any()
    only=NegativeSampler(4,torch.ones(4),kind).sample(torch.tensor([2,3]),10)
    assert (only[0]==3).all() and (only[1]==2).all()


def test_causal_padding_empty():
    encoder=SequenceEncoder(10,dim=8,max_len=4,heads=2,layers=1,dropout=0).eval()
    a=torch.tensor([[2,3,4,0],[0,0,0,0]])
    b=torch.tensor([[2,3,5,0],[0,0,0,0]])
    assert torch.isfinite(encoder(a)).all()
    torch.testing.assert_close(encoder.sequence(a)[:,:3],encoder.sequence(b)[:,:3])
    torch.testing.assert_close(encoder(torch.tensor([[2,3]])),encoder(torch.tensor([[2,3,0,0]])))


@pytest.mark.parametrize('kind',['sasrec','gru4rec'])
@pytest.mark.parametrize('loss',['sampled_binary','full_softmax'])
def test_retrieval_forward_loss(prepared,kind,loss):
    manifest,stats,_=load_artifacts(prepared['data']['prepared'])
    config=deepcopy(prepared)
    config['retrieval'].update(model=kind,loss=loss,max_len=manifest['max_len'])
    model=NeuralRetriever(manifest['n_items'],config['retrieval'])
    batch=next(iter(loader(ResearchDataset(config['data']['prepared'],'train'),config)))
    sampler=NegativeSampler(manifest['n_items'],torch.ones(manifest['n_items']))
    numerator,count=components(model,batch,config,'retrieval',sampler)['retrieval']
    assert count>0 and torch.isfinite(numerator)
    numerator.backward()
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    batch['label'].fill_(0)
    zero,_=components(model,batch,config,'retrieval',sampler)['retrieval']
    assert zero.item()==0 and zero.requires_grad
    _,ids=exact_topk(model.eval(),batch['history'],3,chunk=2)
    dense=model(batch['history'])@model.item_vectors(torch.arange(2,manifest['n_items'])).T/(model.dim**.5)
    expected=dense.argsort(-1,descending=True,stable=True)[:,:3]+2
    assert torch.equal(ids,expected)


@pytest.mark.parametrize('kind',['dnn','din','shared_bottom','mmoe'])
@pytest.mark.parametrize('mode',['raw','log1p','normalized','bucket','cread_inspired','duration_conditioned'])
def test_ranking_shapes_gradients_masks(prepared,kind,mode):
    config=deepcopy(prepared)
    config['ranking']['model']=kind
    config['watchtime']['mode']=mode
    manifest,stats,_=load_artifacts(config['data']['prepared'])
    data=ResearchDataset(config['data']['prepared'],'train')
    watch=fit_watchtime(data,config['watchtime'])
    model=RankingNetwork(manifest,config['ranking'],watch)
    batch=transform_batch(next(iter(loader(data,config))),stats)
    output=model(batch)
    assert output['binary'].shape==batch['label'].shape==output['watch'].shape
    assert 'conversion' not in output and 'ctcvr' not in output
    assert torch.isfinite(output['watch']).all()
    terms=components(model,batch,config,'ranking')
    sum(num/count.clamp_min(1) for num,count in terms.values()).backward()
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    batch['click'].fill_(0)
    assert not masks(batch,config,'ranking')['watch'].any()
    batch['dwell'].fill_(-1)
    assert not masks(batch,config,'ranking')['watch'].any()


def test_esmm_and_no_conversion(prepared):
    config=deepcopy(prepared)
    config['ranking']['tasks']['conversion']=1
    manifest,stats,_=load_artifacts(config['data']['prepared'])
    data=ResearchDataset(config['data']['prepared'],'train')
    watch=fit_watchtime(data,config['watchtime'])
    with pytest.raises(ValueError,match='No observed conversion'):
        RankingNetwork(manifest,config['ranking'],watch)
    manifest['train_labels']['conversion']=4
    model=RankingNetwork(manifest,config['ranking'],watch)
    batch=transform_batch(next(iter(loader(data,config))),stats)
    output=model(batch)
    torch.testing.assert_close(output['ctcvr'],output['binary']*output['conversion'])
    assert masks(batch,config,'ranking')['conversion'].sum()==0
    batch['conversion']=torch.where(batch['click']==1,torch.ones_like(batch['click']),torch.zeros_like(batch['click']))
    numerator,count=components(model,batch,config,'ranking')['conversion']
    assert count==len(batch['click']) and torch.isfinite(numerator)


def test_discretization_restoration():
    edges,_=adaptive_edges(np.array([0,0,1,1,10,30]),4,100,60)
    assert edges[0]==0 and edges[-1]==60 and (np.diff(edges)>0).all()
    artifact={'mode':'cread_inspired','cap_seconds':60,'edges':[0,10,60],
              'centers':[5,35],'duration_edges':[],'duration_mu':[1.],'duration_sigma':[1.],'spec':{}}
    head=WatchHead(4,artifact)
    with torch.no_grad():
        head.project.weight.zero_()
        head.project.bias.zero_()
    output=head(torch.ones(3,4),torch.ones(3))
    torch.testing.assert_close(output['watch'],torch.full((3,),30.))
    loss=head.per_sample_loss(output,{'dwell':torch.tensor([0.,10.,60.]),'duration':torch.ones(3)})
    assert torch.isfinite(loss).all()


def test_diagnostics_do_not_pollute_grad(prepared):
    manifest,stats,_=load_artifacts(prepared['data']['prepared'])
    data=ResearchDataset(prepared['data']['prepared'],'train')
    model=RankingNetwork(manifest,prepared['ranking'],fit_watchtime(data,prepared['watchtime']))
    for p in model.parameters():
        p.grad=torch.ones_like(p)
    before=[p.grad.clone() for p in model.parameters()]
    result=task_gradients(model,transform_batch(next(iter(loader(data,prepared))),stats),prepared)
    assert 'binary:watch' in result['gradient_cosines'] and model.training
    for p,g in zip(model.parameters(),before):
        torch.testing.assert_close(p.grad,g)


@pytest.mark.parametrize('kind',['popularity','itemknn'])
def test_count_baseline(prepared,kind):
    data=ResearchDataset(prepared['data']['prepared'],'train')
    baseline=CatalogueBaseline(fit_baseline(data,{'model':kind,'knn_neighbors':2,'knn_block':2}))
    scores,ids=exact_topk(baseline,torch.tensor([[0,0,0,0],[2,3,0,0]]),3,2)
    assert scores.shape==ids.shape==(2,3) and (ids>=2).all()
