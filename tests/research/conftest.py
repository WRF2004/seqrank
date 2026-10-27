"""Deterministic synthetic fixtures isolated from all real-data entry points."""
import csv
from pathlib import Path

import pytest

from seqrank.research.adapters import FIELDS
from seqrank.research.config import load_config
from seqrank.research.dataset import prepare_data


@pytest.fixture
def research_config(tmp_path):
    config=load_config(Path(__file__).parents[2]/'configs/cpu_smoke.yaml')
    source=tmp_path/'events.csv'
    config['data'].update(source=str(source),prepared=str(tmp_path/'prepared'))
    config['run']['output']=str(tmp_path/'run')
    config['evaluation']['output']=str(tmp_path/'eval')
    config['quality']={'output':str(tmp_path/'quality'),'sample_size':30,'plots':False}
    config['training'].update(batch_size=8,epochs=2,lr=.001)
    config['watchtime'].update(cap_seconds=60)
    with source.open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=FIELDS)
        writer.writeheader()
        for user in range(4):
            for step,t in enumerate(range(10000,400000,20000)):
                label=int((user+step)%3!=0)
                writer.writerow(dict(user_id=str(user),item_id=str(step%5),timestamp=t,
                    click=label,valid_play=label,dwell=float(1+step%9) if label else 0,
                    duration=5+step%5,conversion=-1,category=str(step%2),tab='test',policy='synthetic',
                    available_at=t+10000))
        writer.writerow(dict(user_id='cold',item_id='new',timestamp=330000,click=1,valid_play=1,
                             dwell=3,duration=8,conversion=-1,category='new',available_at=334000))
    return config


@pytest.fixture
def prepared(research_config):
    prepare_data(research_config)
    return research_config
