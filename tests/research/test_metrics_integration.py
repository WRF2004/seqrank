from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest
import torch
from fastapi.testclient import TestClient
from sklearn.metrics import average_precision_score, roc_auc_score

from seqrank.research.config import file_hash
from seqrank.research.dataset import load_artifacts
from seqrank.research.evaluation import evaluate
from seqrank.research.metrics import MetricStore, RetrievalMetrics
from seqrank.research.quality import quality_report
from seqrank.research.serving import Bundle, Request, create_app
from seqrank.research.training import load_checkpoint, train


def test_exact_auc_ap_unsorted_gauc(tmp_path):
    store=MetricStore(tmp_path/'scores.db')
    users=np.array([1,2,1,3,2,1])
    y=np.array([0,1,1,0,0,0])
    p=np.array([.1,.5,.5,.9,.5,.1])
    store.binary('click',users[:3],y[:3],p[:3])
    store.binary('click',users[3:],y[3:],p[3:])
    result=store.result()['binary']['click']
    assert result['auc']==pytest.approx(roc_auc_score(y,p))
    assert result['average_precision']==pytest.approx(average_precision_score(y,p))
    assert result['gauc']==pytest.approx((3*1+2*.5)/5)
    assert result['gauc_eligible_users']==2
    store.close()


def test_retrieval_known_ranks():
    metric=RetrievalMetrics([1,3],8)
    metric.update([2,3,1],[[2,4,5],[4,3,2],[2,3,4]])
    result=metric.result()
    assert result['1']['recall']==pytest.approx(1/3)
    assert result['3']['ndcg']==pytest.approx((1+1/np.log2(3))/3)
    assert result['3']['mrr']==pytest.approx(.5)
    assert result['3']['coverage']==pytest.approx(4/6)


def test_request_validation():
    with pytest.raises(ValueError):
        Request(user_id='u',timestamp=10,dataset_version='x',k=11,candidates=10)
    with pytest.raises(ValueError):
        Request(user_id='u',timestamp=10,dataset_version='x',history=[
        {'item_id':'i','timestamp':1,'available_at':10}])


def test_cpu_full_pipeline(prepared):
    """Executes only when user runs pytest; no assumed model-quality improvements."""
    config=deepcopy(prepared)
    config['training']['epochs']=1
    config['watchtime']['mode']='cread_inspired'
    for stage in ('retrieval','ranking'):
        train(config,stage)
        for split in ('valid','test'):
            result=evaluate(config,stage,split)
            assert result['split']==split and result['stage']==stage
    result=evaluate(config,'e2e','test')
    assert result['metrics']['all']['count']>0
    quality=quality_report(config)
    assert quality['splits']['test']['oov_rate']['items']>0
    bundle=Bundle(config)
    request=Request(user_id='0',timestamp=400000,dataset_version=bundle.version,k=2,candidates=4)
    response=bundle.recommend([request])[0]
    assert len(response['items'])==2
    client=TestClient(create_app(config,bundle))
    assert client.post('/recommend',json=request.model_dump()).status_code==200
    assert client.post('/recommend',json={**request.model_dump(),'dataset_version':'wrong'}).status_code==422
    assert client.post('/recommend',json={**request.model_dump(),'k':0}).status_code==422
    path=Path(config['run']['output'])/'ranking/best.pt'
    model,state=load_checkpoint(path,'cpu',bundle.version)
    assert state['watch_artifact']['mode']=='cread_inspired'
    with pytest.raises(ValueError,match='fingerprint'):
        load_checkpoint(path,'cpu','wrong')


def test_epoch_resume_matches_uninterrupted(prepared,tmp_path):
    config=deepcopy(prepared)
    config['training']['epochs']=2
    config['run']['output']=str(tmp_path/'full')
    train(config,'ranking')
    expected=torch.load(tmp_path/'full/ranking/last.pt',weights_only=True)
    config['run']['output']=str(tmp_path/'resumed')
    train(config,'ranking',stop_after=1)
    last=tmp_path/'resumed/ranking/last.pt'
    train(config,'ranking',resume=str(last))
    actual=torch.load(last,weights_only=True)
    assert expected['epoch']==actual['epoch']==1
    assert expected['step']==actual['step']
    for key in expected['model']:
        torch.testing.assert_close(expected['model'][key],actual['model'][key],rtol=1e-6,atol=1e-7)
    assert 'optimizer' in actual and 'scheduler' in actual and 'scaler' in actual and len(actual['rng'])==1


def test_batch_inference_consistency(prepared):
    config=deepcopy(prepared)
    config['training']['epochs']=1
    train(config,'ranking')
    train(config,'retrieval')
    bundle=Bundle(config)
    small=Request(user_id='0',timestamp=400000,dataset_version=bundle.version,candidates=2,k=2)
    large=Request(user_id='1',timestamp=400000,dataset_version=bundle.version,candidates=5,k=2)
    alone=bundle.recommend([small])[0]['items']
    together=bundle.recommend([small,large])[0]['items']
    assert [x['item_id'] for x in alone]==[x['item_id'] for x in together]
    np.testing.assert_allclose([x['score'] for x in alone],[x['score'] for x in together],rtol=1e-5)


def test_ann_version_rejection(prepared,tmp_path):
    pytest.importorskip('faiss')
    from seqrank.research.retrievers import ANN, build_index
    config=deepcopy(prepared)
    config['training']['epochs']=1
    train(config,'retrieval')
    path=Path(config['run']['output'])/'retrieval/best.pt'
    manifest,_,_=load_artifacts(config['data']['prepared'])
    model,state=load_checkpoint(path)
    index=tmp_path/'hnsw.faiss'
    build_index(model,index,file_hash(path),manifest['fingerprint'],{})
    with pytest.raises(ValueError,match='identity'):
        ANN(index,'wrong',manifest['fingerprint'])
