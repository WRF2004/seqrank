"""Fail-closed semantic contracts and an explicit final-test freeze artifact."""
from pathlib import Path
import json

from .config import fingerprint, file_hash, write_json


def evaluation_protocol(config):
    spec=config.get('evaluation',{})
    return {key:spec.get(key,default) for key,default in {
        'ks':[10,50,100], 'candidates':100, 'exclude_seen':False, 'tail_quantile':.8}.items()}


def check_configuration(config,state,manifest):
    if state['manifest']!=manifest:
        raise ValueError('Checkpoint data manifest/protocol differs from loaded data')
    for key in ('primary_label','history_protocol','max_len','train_end','valid_end'):
        if key in config['data']:
            from .dataset import timestamp_ms
            value=config['data'][key]
            if key.endswith('_end'):
                value=timestamp_ms(value)
            if value!=manifest[key]:
                raise ValueError(f'Data protocol mismatch: {key}')
    statistics=json.loads((Path(config['data']['prepared'])/'features.json').read_text())
    if state['statistics']!=statistics:
        raise ValueError('Checkpoint feature statistics mismatch')
    if state['stage']=='ranking' and state['config']['ranking']['tasks'].get('watch',0)>0:
        watch=state.get('watch_artifact')
        if not watch or watch.get('fit_split')!='train' or watch.get('spec')!=state['config']['watchtime']:
            raise ValueError('Checkpoint watch transform mismatch')
    sections=['retrieval'] if state['stage']=='retrieval' else ['ranking','watchtime']
    for section in sections:
        if config.get(section)!=state['config'].get(section):
            raise ValueError(f'Checkpoint/config mismatch: {section}')


def lock_path(config):
    return Path(config.get('evaluation',{}).get('lock',str(Path(config['run']['output'])/'final-evaluation.json')))


def freeze(config):
    from .dataset import load_artifacts
    from .evaluation import checkpoint_path
    from .training import load_checkpoint
    manifest,_,_=load_artifacts(config['data']['prepared'])
    checkpoints={}
    for stage in ('retrieval','ranking'):
        path=Path(checkpoint_path(config,stage))
        if path.is_file():
            _,state=load_checkpoint(path,dataset_hash=manifest['fingerprint'])
            check_configuration(config,state,manifest)
            if state['stage']!=stage:
                raise ValueError('Checkpoint stage mismatch')
            checkpoints[stage]=file_hash(path)
    if not checkpoints:
        raise ValueError('Freeze requires at least one selected trained checkpoint')
    value={'dataset':manifest['fingerprint'],'checkpoints':checkpoints,'protocol':evaluation_protocol(config)}
    value['fingerprint']=fingerprint(value)
    path=lock_path(config)
    if path.exists():
        raise FileExistsError(f'Final protocol already frozen: {path}')
    write_json(path,value)
    return value


def require_final(config,manifest,stage,path):
    lock=json.loads(lock_path(config).read_text())
    identity=dict(lock)
    expected=identity.pop('fingerprint')
    if fingerprint(identity)!=expected or lock['dataset']!=manifest['fingerprint'] or lock['protocol']!=evaluation_protocol(config):
        raise ValueError('Final evaluation protocol changed after freeze')
    if lock['checkpoints'].get(stage)!=file_hash(path):
        raise ValueError('Selected checkpoint changed or was not frozen')
