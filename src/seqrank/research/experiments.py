"""Traceable experiment matrices and measured-results-only aggregation."""
from __future__ import annotations

import csv
from copy import deepcopy
import glob
import json
from pathlib import Path

import numpy as np

from .config import load_config, merge, write_json
from .evaluation import evaluate
from .training import train


def matrix(config,execute=False):
    spec=config['experiments']
    root=Path(spec['output'])
    plans=[]
    base=load_config(spec['base'])
    for name,change in spec['variants'].items():
        if '/' in name or '..' in name:
            raise ValueError('Variant names must be simple directory names')
        for seed in spec.get('seeds',[42]):
            run=deepcopy(merge(base,change))
            run['seed']=int(seed)
            run['run']={'output':str(root/name/str(seed))}
            run.setdefault('evaluation',{})['output']=str(root/name/str(seed)/'evaluation')
            run['experiment']={'variant':name,'overrides':change,'seed':seed}
            target=root/name/str(seed)/'resolved.json'
            if target.exists():
                existing=json.loads(target.read_text())
                if existing!=run:
                    raise ValueError(f'Existing experiment configuration differs: {target}')
            else:
                write_json(target,run)
            plans.append(str(target))
            if execute:
                for stage in spec.get('stages',['ranking']):
                    train(run,stage)
                    evaluate(run,stage,'valid')
    write_json(root/'plan.json',{'executed':execute,'configs':plans})
    return {'executed':execute,'configs':plans}


def flatten(value,prefix=''):
    for key,item in value.items():
        name=f'{prefix}.{key}' if prefix else key
        if isinstance(item,dict):
            yield from flatten(item,name)
        elif isinstance(item,(int,float)) and not isinstance(item,bool):
            yield name,float(item)


def report(config):
    spec=config['report']
    records={}
    seen=set()
    sources=sorted({p for pattern in spec['sources'] for p in glob.glob(pattern,recursive=True)})
    if not sources:
        raise FileNotFoundError('No measured metrics.json files found')
    for path in sources:
        row=json.loads(Path(path).read_text())
        if 'metrics' not in row or 'checkpoint_hash' not in row:
            raise ValueError(f'Not a measured result: {path}')
        run=row['config']
        variant=run.get('experiment',{}).get('variant',run['run']['output'])
        identity=(variant,row['stage'],row['split'],row['dataset_hash'],row['protocol']+':'+row.get('evaluation_protocol_fingerprint','legacy-unbound'))
        duplicate=(identity,row['seed'])
        if duplicate in seen:
            raise ValueError(f'Duplicate seed/result: {path}')
        seen.add(duplicate)
        for metric,value in flatten(row['metrics']):
            records.setdefault(identity+(metric,),[]).append(value)
    output=Path(spec['output'])
    output.mkdir(parents=True,exist_ok=True)
    rows=[]
    for key,values in sorted(records.items()):
        variant,stage,split,dataset,protocol,metric=key
        rows.append(dict(variant=variant,stage=stage,split=split,dataset=dataset,protocol=protocol,metric=metric,
                         n=len(values),mean=float(np.mean(values)),std=float(np.std(values,ddof=1)) if len(values)>1 else None))
    write_json(output/'comparison.json',{'sources':sources,'rows':rows,'std_definition':'sample standard deviation; null for n=1'})
    with (output/'comparison.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]) if rows else ['metric'])
        writer.writeheader()
        writer.writerows(rows)
    lines=['# Measured experiment comparison','', '| Variant | Stage | Split | Metric | Seeds | Mean | Sample std |',
           '|---|---|---|---|---:|---:|---:|']
    lines += [f"| {r['variant']} | {r['stage']} | {r['split']} | {r['metric']} | {r['n']} | {r['mean']:.6g} | {r['std'] if r['std'] is not None else '—'} |" for r in rows]
    lines += ['', 'Compare only matching dataset hashes and protocols (see CSV).', '',
              '## Interpretation to complete after inspecting actual results',
              '- Validation selection and test isolation:', '- Task trade-offs and seed variability:',
              '- Duration-group errors and calibration:', '- Latency/resource trade-offs:',
              '- Limitations; no offline-to-online uplift inference:']
    (output/'comparison.md').write_text('\n'.join(lines)+'\n')
    return {'output':str(output),'rows':len(rows)}
