import csv
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from seqrank.research.adapters import adapt_row, convert, inspect
from seqrank.research.dataset import ResearchDataset, TemporalSplitter, load_artifacts, prepare_data
from seqrank.research.features import dense_features
from seqrank.research.watchtime import fit_watchtime


def test_kuairand_semantics_and_units():
    row={'user_id':'1','video_id':'2','time_ms':'1000','is_click':'1','play_time_ms':'8000','duration_ms':'10000','tab':'1'}
    result=adapt_row(row,{'kind':'kuairand'},Path('log_standard_1k.csv'))
    assert result['click']==-1 and result['valid_play']==1 and result['dwell']==8 and result['conversion']==-1
    with pytest.raises(ValueError,match='verified'):
        adapt_row(row,{'kind':'kuairand','primary_label':'click'},Path('log_standard.csv'))
    result=adapt_row(row,{'kind':'kuairand','primary_label':'click','tab_semantics':{'1':'click'}},Path('log_standard.csv'))
    assert result['click']==1


def test_clipping_does_not_advance_feedback():
    row={'user_id':'u','item_id':'i','timestamp':100,'click':1,'dwell':100,'duration':10}
    result=adapt_row(row,{'kind':'generic','loop_policy':'clip','outlier_policy':'clip','max_watch_seconds':20},Path('x.csv'))
    assert result['dwell']==10 and result['available_at']==100100


def test_schema_missing_column(tmp_path):
    path=tmp_path/'log.csv'
    path.write_text('user_id,time_ms\n1,2\n')
    with pytest.raises(ValueError,match='missing columns'):
        inspect({'adapter':{'kind':'kuairand','sources':[str(path)]}})


def test_split_and_train_only_vocab(prepared):
    manifest,stats,_=load_artifacts(prepared['data']['prepared'])
    assert manifest['train_labels']['conversion']==0
    assert manifest['oov']['test']['users']==1 and manifest['oov']['test']['items']==1
    assert stats['fit_split']=='train'
    splitter=TemporalSplitter(200000,300000)
    assert [splitter(t) for t in [199999,200000,299999,300000]]==['train','valid','valid','test']
    dataset=ResearchDataset(prepared['data']['prepared'],'train')
    dense=torch.stack([dataset[i]['dense'] for i in range(len(dataset))]).numpy()
    np.testing.assert_allclose(dense.mean(0),stats['dense_mean'],atol=1e-5)
    with pytest.raises(ValueError,match='train shards'):
        fit_watchtime(ResearchDataset(prepared['data']['prepared'],'test'),prepared['watchtime'])


def test_same_timestamp_and_delayed_history(research_config):
    source=Path(research_config['data']['source'])
    with source.open() as f:
        records=list(csv.DictReader(f))
    # Same user/time: neither can consume the other's feedback. Delayed older event
    # must not displace more recent events when finally available.
    special=[]
    for item,t,a in [('a',1000,100000),('b',1000,1001),('c',2000,2001),('d',3000,3001),('e',110000,110001)]:
        special.append(dict(user_id='special',item_id=item,timestamp=t,available_at=a,click=1,
                            valid_play=1,dwell=1,duration=5,conversion=-1,category='x'))
    records+=special
    with source.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(records[0]))
        w.writeheader()
        w.writerows(records)
    research_config['data']['max_len']=2
    prepare_data(research_config)
    dataset=ResearchDataset(research_config['data']['prepared'],'train')
    early=[dataset[i] for i in range(len(dataset)) if int(dataset[i]['timestamp'])==1000]
    assert len(early)==2 and all(not r['history'].any() for r in early)
    later=[dataset[i] for i in range(len(dataset)) if int(dataset[i]['timestamp'])==110000 and int((dataset[i]['history_time']==3000).sum())]
    assert len(later)==1
    assert later[0]['history_time'].tolist()==[2000,3000]


def test_history_is_past_and_recomputed(prepared):
    for split in ('train','valid','test'):
        dataset=ResearchDataset(prepared['data']['prepared'],split)
        for i in range(len(dataset)):
            r=dataset[i]
            m=r['history']!=0
            assert (r['history_time'][m]<r['timestamp']).all()
            x=dense_features(r['duration'].numpy()[None],r['timestamp'].numpy()[None],
                *[r[k].numpy()[None] for k in ['history','history_watch','history_duration','history_time']])
            np.testing.assert_allclose(x[0],r['dense'].numpy())


def test_manifest_tamper_rejected(prepared):
    path=Path(prepared['data']['prepared'])/'manifest.json'
    m=json.loads(path.read_text())
    m['max_len']+=1
    path.write_text(json.dumps(m))
    with pytest.raises(ValueError,match='fingerprint'):
        load_artifacts(path.parent)


def test_frozen_protocol(research_config):
    research_config['data']['history_protocol']='frozen_train'
    prepare_data(research_config)
    dataset=ResearchDataset(research_config['data']['prepared'],'test')
    for i in range(len(dataset)):
        assert (dataset[i]['history_time']<research_config['data']['train_end']).all()


def test_adapter_roundtrip_and_content(tmp_path):
    source=tmp_path/'log_standard_pure.csv'
    source.write_text('user_id,video_id,time_ms,is_click,play_time_ms,tab\nu,i,1000,1,8000,1\n')
    content=tmp_path/'content.csv'
    content.write_text('video_id,video_duration,tag\ni,10000,cat\n')
    config={'adapter':{'kind':'kuairand','sources':[str(source)],'output':str(tmp_path/'out.csv'),
        'content':{'sources':[str(content)],'assert_immutable':True}}}
    metadata=convert(config)
    with (tmp_path/'out.csv').open() as f:
        row=next(csv.DictReader(f))
    assert row['category']=='cat' and float(row['duration'])==10 and row['valid_play']=='1'
    assert metadata['rows']==1
