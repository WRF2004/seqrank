"""Generate observed quality reports; no embedded real-data statistics."""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from .config import empty_output, write_json
from .dataset import ResearchDataset, load_artifacts


def summarize(values):
    x = np.asarray(values,dtype=float)
    return {'n':len(x),'mean':float(x.mean()) if len(x) else None,
            'quantiles':{str(q):float(np.quantile(x,q)) for q in [0,.25,.5,.75,.9,.99,1]} if len(x) else {}}


def quality_report(config):
    root = config['data']['prepared']
    manifest,_,_ = load_artifacts(root)
    output = empty_output(config['quality']['output'])
    rng = np.random.default_rng(config.get('seed',42))
    limit = int(config['quality'].get('sample_size',100000))
    if limit<1:
        raise ValueError('sample_size must be positive')
    report = {'dataset':manifest['fingerprint'],'sampling':'uniform reservoir for distributions; missing/OOV/counts exact',
              'cardinalities':manifest['cardinalities'],'splits':{},'train_users':manifest['n_users']-2,'train_items':manifest['n_items']-2}
    chart_data = {}
    for split in ('train','valid','test'):
        data = ResearchDataset(root,split)
        sample=[]
        missing={k:0 for k in ['label','click','conversion','dwell','duration']}
        positives=negatives=0
        for i in range(len(data)):
            row = data[i]
            for k in missing:
                missing[k]+=int(row[k]<0)
            positives+=int(row['label']==1)
            negatives+=int(row['label']==0)
            point = [float(row['dwell']),float(row['duration']),int((row['history']!=0).sum())]
            if len(sample)<limit:
                sample.append(point)
            else:
                j=int(rng.integers(i+1))
                if j<limit:
                    sample[j]=point
        x=np.array(sample)
        distributions={k:summarize(x[x[:,j]>=0,j]) for j,k in enumerate(['watch_seconds','duration_seconds','history_length'])}
        report['splits'][split]={'rows':len(data),'positive':positives,'negative':negatives,
            'positive_rate':positives/max(1,positives+negatives),'missing_rate':{k:v/len(data) for k,v in missing.items()},
            'oov_rate':{k:v/len(data) for k,v in manifest['oov'][split].items()},'distributions':distributions}
        chart_data[split]=x
    for name in ['activity','popularity']:
        report['training_'+name]=summarize(np.load(Path(root)/f'{name}.npy')[2:])
    write_json(output/'quality.json',report)
    with (output/'splits.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=['split','rows','positive','negative','positive_rate'])
        writer.writeheader()
        for split,values in report['splits'].items():
            writer.writerow({'split':split,**{k:values[k] for k in writer.fieldnames if k!='split'}})
    from .experiments import flatten
    with (output/'statistics.csv').open('w',newline='') as f:
        writer=csv.writer(f)
        writer.writerow(['metric','value'])
        writer.writerows(flatten(report))
    if config['quality'].get('plots',False):
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        for j,name in enumerate(['watch_seconds','duration_seconds','history_length']):
            fig,ax=plt.subplots()
            for split,x in chart_data.items():
                ax.hist(x[x[:,j]>=0,j],bins=50,alpha=.4,label=split)
            ax.set(xlabel=name,ylabel='sample count')
            ax.legend()
            fig.savefig(output/f'{name}.png',dpi=160,bbox_inches='tight')
            plt.close(fig)
    return report
