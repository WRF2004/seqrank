"""Disk-backed exact binary metrics and bounded-memory retrieval accumulators."""
from __future__ import annotations

import math
import sqlite3
from pathlib import Path

import numpy as np


class BinaryAccumulator:
    """Histogram metrics with explicit bin count; exact metrics use MetricStore."""
    def __init__(self,bins=4096,ece_bins=20):
        self.pos = np.zeros(bins)
        self.neg = np.zeros(bins)
        self.ece_n = np.zeros(ece_bins)
        self.ece_y = np.zeros(ece_bins)
        self.ece_p = np.zeros(ece_bins)
        self.loss = 0.

    def update(self,y,p):
        y,p = np.asarray(y),np.asarray(p,dtype=float)
        mask = y>=0
        y,p = y[mask],p[mask]
        if not np.isfinite(p).all() or ((p<0)|(p>1)).any() or not np.isin(y,[0,1]).all():
            raise ValueError('Invalid binary metric input')
        index = np.minimum((p*len(self.pos)).astype(int),len(self.pos)-1)
        np.add.at(self.pos,index,y)
        np.add.at(self.neg,index,1-y)
        index = np.minimum((p*len(self.ece_n)).astype(int),len(self.ece_n)-1)
        np.add.at(self.ece_n,index,1)
        np.add.at(self.ece_y,index,y)
        np.add.at(self.ece_p,index,p)
        q = p.clip(1e-7,1-1e-7)
        self.loss += float((-y*np.log(q)-(1-y)*np.log1p(-q)).sum())

    def result(self):
        pos,neg = self.pos.sum(),self.neg.sum()
        n = pos+neg
        tp,fp = np.cumsum(self.pos[::-1]),np.cumsum(self.neg[::-1])
        return {'count':int(n),'positive':int(pos),
                'auc':float((self.pos*(np.cumsum(self.neg)-self.neg/2)).sum()/(pos*neg)) if pos and neg else None,
                'average_precision':float((self.pos[::-1]*tp/np.maximum(tp+fp,1)).sum()/pos) if pos else None,
                'method':f'histogram_{len(self.pos)}_bins_ties_half_credit',
                'logloss':self.loss/n if n else None,
                'ece':float(np.abs(self.ece_y-self.ece_p).sum()/n) if n else None,
                'ece_bins':len(self.ece_n)}


class MetricStore:
    """SQLite external sorts give exact AUC/AP/GAUC without keeping all predictions in RAM."""
    def __init__(self,path):
        path = Path(path)
        if path.exists():
            raise FileExistsError(path)
        self.db = sqlite3.connect(path)
        self.db.execute('PRAGMA temp_store=FILE')
        self.db.execute('CREATE TABLE scores(task TEXT,u INTEGER,y INTEGER,p REAL)')
        self.db.execute('CREATE TABLE watch(u INTEGER,y REAL,p REAL,d REAL,a REAL,pref REAL,bucket INTEGER)')
        self.accumulators = {}

    def binary(self,task,user,y,p):
        mask = np.asarray(y)>=0
        user,y,p = np.asarray(user)[mask],np.asarray(y)[mask],np.asarray(p)[mask]
        metric = self.accumulators.setdefault(task,BinaryAccumulator())
        metric.update(y,p)
        self.db.executemany('INSERT INTO scores VALUES(?,?,?,?)',
                            ((task,int(u),int(label),float(score)) for u,label,score in zip(user,y,p)))

    def watch(self,user,y,p,duration,activity,preference,bucket):
        self.db.executemany('INSERT INTO watch VALUES(?,?,?,?,?,?,?)',
            ((int(u),float(y0),float(p0),float(d),float(a),float(h),int(b))
             for u,y0,p0,d,a,h,b in zip(user,y,p,duration,activity,preference,bucket)))

    def binary_result(self,task):
        result = self.accumulators[task].result()
        pos,neg = result['positive'],result['count']-result['positive']
        neg_below,pairs = 0.,0.
        for p,positive,negative in self.db.execute('SELECT p,sum(y),count(*)-sum(y) FROM scores WHERE task=? GROUP BY p ORDER BY p',(task,)):
            pairs += positive*(neg_below+.5*negative)
            neg_below += negative
        result['auc'] = pairs/(pos*neg) if pos and neg else None
        tp,fp,ap = 0.,0.,0.
        for p,positive,negative in self.db.execute('SELECT p,sum(y),count(*)-sum(y) FROM scores WHERE task=? GROUP BY p ORDER BY p DESC',(task,)):
            tp += positive
            fp += negative
            ap += positive*tp/(tp+fp)
        result['average_precision'] = ap/pos if pos else None
        result['pr_auc_definition'] = 'average_precision_step_integral; not trapezoidal PR area'
        result['method'] = 'exact_sql_grouped_scores_ties_half_credit'
        current = None
        pn=nn=area=0.
        weighted=eligible=groups=0
        def finish(pn,nn,area):
            return ((area/(pn*nn))*(pn+nn),int(pn+nn),1) if pn and nn else (0.,0,0)
        for user,p,positive,negative in self.db.execute('SELECT u,p,sum(y),count(*)-sum(y) FROM scores WHERE task=? GROUP BY u,p ORDER BY u,p',(task,)):
            if current is not None and user!=current:
                w,n,g = finish(pn,nn,area)
                weighted+=w
                eligible+=n
                groups+=g
                pn=nn=area=0.
            current = user
            area += positive*(nn+.5*negative)
            pn+=positive
            nn+=negative
        w,n,g = finish(pn,nn,area)
        weighted+=w
        eligible+=n
        groups+=g
        result.update(gauc=weighted/eligible if eligible else None,gauc_eligible_rows=eligible,
                      gauc_eligible_users=groups,gauc_weight='eligible impression count; single-class users excluded')
        return result

    @staticmethod
    def regression(values):
        n=0
        absolute=squared=logabsolute=target=prediction=0.
        for y,p in values:
            n+=1
            absolute+=abs(y-p)
            squared+=(y-p)**2
            logabsolute+=abs(math.log1p(y)-math.log1p(p))
            target+=y
            prediction+=p
        return {'count':n,'mae':absolute/n if n else None,'rmse':math.sqrt(squared/n) if n else None,
                'log_mae':logabsolute/n if n else None,'mean_target':target/n if n else None,
                'mean_prediction':prediction/n if n else None,'mean_signed_error':(prediction-target)/n if n else None}

    def result(self,duration_edges=(),watch_edges=()):
        self.db.commit()
        result = {'binary':{task:self.binary_result(task) for task in self.accumulators}}
        result['watch'] = self.regression(self.db.execute('SELECT y,p FROM watch'))
        groups = {}
        for field,edges in [('d',duration_edges),('a',[1,5,10,20,50]),('pref',[0,1,2,3,4])]:
            expression = '+'.join(f'({field}>={float(e)})' for e in edges) or '0'
            if field=='d':
                expression = f'CASE WHEN d<=0 THEN -1 ELSE {expression} END'
            names = [r[0] for r in self.db.execute(f'SELECT DISTINCT {expression} FROM watch')]
            groups[field] = {str(group):self.regression(self.db.execute(f'SELECT y,p FROM watch WHERE ({expression})=?',(group,))) for group in names}
            groups[field]['edges'] = list(edges)
        groups['target_bucket'] = {str(group):self.regression(self.db.execute('SELECT y,p FROM watch WHERE bucket=?',(group,))) for (group,) in self.db.execute('SELECT DISTINCT bucket FROM watch')}
        # Descriptive long/short score and residual gaps, not causal debiasing claims.
        result['watch_groups'] = groups
        predicted_bin = '+'.join(f'(p>={float(e)})' for e in watch_edges) or '0'
        result['watch_calibration'] = {str(group):self.regression(self.db.execute(
            f'SELECT y,p FROM watch WHERE ({predicted_bin})=?',(group,)))
            for (group,) in self.db.execute(f'SELECT DISTINCT {predicted_bin} FROM watch')}
        result['watch_calibration_edges'] = list(watch_edges)
        duration_bin = '+'.join(f'(d>={float(e)})' for e in duration_edges) or '0'
        # Within-user percentile ranks with average ties. This is an exposure-only descriptive proxy.
        query = f"""WITH r AS (SELECT d,
            (RANK() OVER(PARTITION BY u ORDER BY p)+(COUNT(*) OVER(PARTITION BY u,p)-1)/2.0-1)
              /MAX(COUNT(*) OVER(PARTITION BY u)-1,1) AS rp,
            (RANK() OVER(PARTITION BY u ORDER BY y)+(COUNT(*) OVER(PARTITION BY u,y)-1)/2.0-1)
              /MAX(COUNT(*) OVER(PARTITION BY u)-1,1) AS ry FROM watch)
            SELECT {duration_bin} AS g,count(*),avg(rp),avg(ry),avg(rp-ry)
            FROM r WHERE d>0 GROUP BY g"""
        result['duration_rank_bias_proxy'] = {str(g):{'count':n,'mean_prediction_percentile':p,
            'mean_observed_percentile':y,'mean_rank_gap':gap} for g,n,p,y,gap in self.db.execute(query)}
        return result

    def close(self):
        self.db.close()


class RetrievalMetrics:
    def __init__(self,ks,n_items):
        self.ks = sorted(set(ks))
        self.n_items = n_items
        self.n = 0
        self.sums = {k:np.zeros(3) for k in self.ks}
        self.covered = {k:set() for k in self.ks}

    def update(self,targets,ids):
        targets,ids = np.asarray(targets),np.asarray(ids)
        self.n += len(targets)
        for k in self.ks:
            top = ids[:,:k]
            match = (top==targets[:,None])&(top>=2)
            ranks = np.arange(1,top.shape[1]+1)
            self.sums[k] += [match.any(1).sum(),(match/np.log2(ranks+1)).sum(),(match/ranks).sum()]
            self.covered[k].update(top[top>=2].tolist())

    def result(self):
        return {'count':self.n,'protocol':'one observed positive per request; unknown targets count as misses',
                **{str(k):{'hit':float(v[0]/self.n) if self.n else None,
                          'recall':float(v[0]/self.n) if self.n else None,
                          'ndcg':float(v[1]/self.n) if self.n else None,
                          'mrr':float(v[2]/self.n) if self.n else None,
                          'coverage':len(self.covered[k])/max(1,self.n_items-2)} for k,v in self.sums.items()}}
