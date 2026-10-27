"""Version-2 disk preprocessing. Train-only fitting, delayed feedback, immutable artifacts."""
from __future__ import annotations

import heapq
import json
import sqlite3
from datetime import datetime
from pathlib import Path

import numpy as np

from ..data import ShardedDataset
from .adapters import binary, number, rows
from .config import empty_output, file_hash, fingerprint, write_json
from .features import DENSE_NAMES, dense_features


def record_dtype(length: int) -> np.dtype:
    return np.dtype([('user', '<i8'), ('group_user', '<i8'), ('item', '<i8'), ('category', '<i8'),
                     ('timestamp', '<i8'), ('history', '<i8', (length,)),
                     ('history_watch', '<f4', (length,)), ('history_duration', '<f4', (length,)),
                     ('history_time', '<i8', (length,)), ('dense', '<f4', (len(DENSE_NAMES),)),
                     ('label', '<f4'), ('click', '<f4'), ('conversion', '<f4'),
                     ('dwell', '<f4'), ('duration', '<f4')])


class ResearchDataset(ShardedDataset):
    def __init__(self, root, split):
        super().__init__(root, split)
        if self.manifest.get('schema_version') != 2:
            raise ValueError('v2 command requires schema_version=2; use prepare-data')


def timestamp_ms(value):
    if isinstance(value, (int, float)) or str(value).isdigit():
        return int(value)
    parsed = datetime.fromisoformat(str(value).replace('Z', '+00:00'))
    if parsed.tzinfo is None:
        raise ValueError('ISO split dates require explicit timezone')
    return int(parsed.timestamp()*1000)


class TemporalSplitter:
    def __init__(self, train_end: int, valid_end: int):
        if not 0 < train_end < valid_end:
            raise ValueError('Require 0 < train_end < valid_end (milliseconds)')
        self.train_end, self.valid_end = train_end, valid_end

    def __call__(self, timestamp: int) -> str:
        return 'train' if timestamp < self.train_end else 'valid' if timestamp < self.valid_end else 'test'


def prepare_data(config: dict) -> dict:
    spec = config['data']
    source = Path(spec['source'])
    if not source.is_file():
        raise FileNotFoundError(f'Missing normalized CSV/Parquet: {source}; run convert first')
    splitter = TemporalSplitter(timestamp_ms(spec['train_end']), timestamp_ms(spec['valid_end']))
    primary = spec.get('primary_label', 'click')
    if primary not in {'click', 'valid_play'}:
        raise ValueError('primary_label must be click or valid_play')
    protocol = spec.get('history_protocol', 'rolling')
    if protocol not in {'rolling', 'frozen_train'}:
        raise ValueError('history_protocol must be rolling or frozen_train')
    length, shard_size = int(spec.get('max_len', 50)), int(spec.get('shard_size', 50000))
    if length < 1 or shard_size < 1:
        raise ValueError('max_len and shard_size must be positive')
    output = empty_output(spec['prepared'])
    db = sqlite3.connect(output / 'events.db')
    db.execute('PRAGMA temp_store=FILE')
    db.execute('CREATE TABLE events(u TEXT,i TEXT,t INTEGER,c REAL,p REAL,v REAL,w REAL,d REAL,cat TEXT,a INTEGER, UNIQUE(u,i,t,c,p,v,w,d,cat,a))')
    duplicate_policy = spec.get('duplicates', 'drop_exact')
    if duplicate_policy not in {'drop_exact', 'keep'}:
        raise ValueError('duplicates must be drop_exact or keep')
    if duplicate_policy == 'keep':
        db.execute('DROP TABLE events')
        db.execute('CREATE TABLE events(u TEXT,i TEXT,t INTEGER,c REAL,p REAL,v REAL,w REAL,d REAL,cat TEXT,a INTEGER)')
    source_count = 0
    for row in rows(source):
        if not {'user_id', 'item_id', 'timestamp'} <= row.keys():
            raise ValueError('Normalized input requires user_id,item_id,timestamp')
        if row['user_id'] is None or row['item_id'] is None or not str(row['user_id']).strip() or not str(row['item_id']).strip():
            raise ValueError('Nonempty user_id and item_id required')
        c, p, v = binary(row.get('click')), binary(row.get('valid_play')), binary(row.get('conversion'))
        w, d, t = number(row.get('dwell')), number(row.get('duration')), int(row['timestamp'])
        if (w < 0 and w != -1) or (d < 0 and d != -1) or t < 0:
            raise ValueError('Negative time/duration other than -1 missing sentinel')
        if v == 1 and c != 1:
            raise ValueError('Conversion requires click=1')
        a = int(number(row.get('available_at')))
        if a < 0:
            a = t + max(1, int(max(w, 0)*1000))
        if a <= t:
            raise ValueError('Feedback must become available strictly after event time')
        if spec.get('missing_primary', 'error') == 'error' and (c if primary == 'click' else p) < 0:
            raise ValueError(f'Missing {primary} label; explicitly set missing_primary=mask to retain')
        db.execute('INSERT OR IGNORE INTO events VALUES(?,?,?,?,?,?,?,?,?,?)',
                   (str(row['user_id']), str(row['item_id']), t, c, p, v, w, d,
                    str(row.get('category') or 'unknown'), a))
        source_count += 1
        if source_count % 10000 == 0:
            db.commit()
    db.commit()
    total = db.execute('SELECT count(*) FROM events').fetchone()[0]
    if not total:
        raise ValueError('Empty dataset')
    db.execute('CREATE INDEX events_order ON events(u,t)')
    for table, column in [('user_vocab','u'), ('item_vocab','i'), ('category_vocab','cat')]:
        db.execute(f'CREATE TABLE {table}(raw TEXT PRIMARY KEY,id INTEGER UNIQUE)')
        values = db.execute(f'SELECT DISTINCT {column} FROM events WHERE t<? ORDER BY {column}', (splitter.train_end,))
        db.executemany(f'INSERT INTO {table} VALUES(?,?)', ((v[0], i+2) for i,v in enumerate(values)))
    db.execute('CREATE TABLE groups(raw TEXT PRIMARY KEY,id INTEGER UNIQUE)')
    values = db.execute('SELECT DISTINCT u FROM events ORDER BY u')
    db.executemany('INSERT INTO groups VALUES(?,?)', ((v[0], i) for i,v in enumerate(values)))
    # groups are evaluation identifiers only, never fed into any model.
    n_users = db.execute('SELECT count(*) FROM user_vocab').fetchone()[0]+2
    n_items = db.execute('SELECT count(*) FROM item_vocab').fetchone()[0]+2
    n_categories = db.execute('SELECT count(*) FROM category_vocab').fetchone()[0]+2
    if n_items < 4:
        raise ValueError('At least two training items required')
    popularity, activity = np.zeros(n_items, np.int64), np.zeros(n_users, np.int64)
    cat = np.ones(n_items, np.int64)
    duration = np.full(n_items, -1, np.float32)
    first_seen = np.full(n_items, np.iinfo(np.int64).max, np.int64)
    query = '''SELECT e.u,COALESCE(uv.id,1),g.id,COALESCE(iv.id,1),COALESCE(cv.id,1),
               e.t,e.c,e.p,e.v,e.w,e.d,e.a FROM events e
               LEFT JOIN user_vocab uv ON uv.raw=e.u LEFT JOIN item_vocab iv ON iv.raw=e.i
               LEFT JOIN category_vocab cv ON cv.raw=e.cat JOIN groups g ON g.raw=e.u
               ORDER BY e.u,e.t,e.rowid'''
    buffers = {s: [] for s in ('train','valid','test')}
    shards = {s: [] for s in buffers}
    counts = {s: 0 for s in buffers}
    availability = {key: 0 for key in ('label','click','conversion','dwell','duration')}
    oov = {s: {'users':0,'items':0,'categories':0} for s in buffers}
    dense_sum, dense_sq, fitted_count = np.zeros(8), np.zeros(8), 0
    previous_user = None
    history, pending = [], []
    sequence = 0
    def flush(split):
        if not buffers[split]:
            return
        name = f'{split}-{len(shards[split]):05d}.npy'
        np.save(output / name, np.array(buffers[split], dtype=record_dtype(length)))
        shards[split].append({'file':name,'rows':len(buffers[split]),'sha256':file_hash(output/name)})
        buffers[split].clear()
    for raw, u, group, item, category, t, click, play, conversion, watch, dur, available in db.execute(query):
        if raw != previous_user:
            history.clear()
            pending.clear()
            previous_user = raw
        # Apply only completed feedback. Frozen protocol never advances beyond train_end.
        horizon = t if protocol == 'rolling' or t < splitter.train_end else splitter.train_end
        while pending and pending[0][0] < horizon:
            _, _, payload = heapq.heappop(pending)
            history.append(payload)
            history.sort(key=lambda x: x[3])
            del history[:-length]
        h = sorted(history, key=lambda x: x[3])[-length:]
        hid = [x[0] for x in h] + [0]*(length-len(h))
        hw = [x[1] for x in h] + [-1]*(length-len(h))
        hd = [x[2] for x in h] + [-1]*(length-len(h))
        ht = [x[3] for x in h] + [0]*(length-len(h))
        dense = dense_features(np.array([dur]), np.array([t]), np.array([hid]),
                               np.array([hw]), np.array([hd]), np.array([ht]))[0]
        label = click if primary == 'click' else play
        split = splitter(t)
        supervised = (label, click, conversion, watch) if split != 'train' or available < splitter.train_end else (-1, -1, -1, -1)
        buffers[split].append((u, group, item, category, t, hid, hw, hd, ht, dense,
                               *supervised, dur))
        counts[split] += 1
        oov[split]['users'] += int(u == 1)
        oov[split]['items'] += int(item == 1)
        oov[split]['categories'] += int(category == 1)
        if split == 'train':
            dense_sum += dense
            dense_sq += dense.astype(np.float64)**2
            fitted_count += 1
            activity[u] += 1
            popularity[item] += int(supervised[0] == 1)
            for key, val in zip(availability, (*supervised, dur)):
                availability[key] += int(val >= 0)
            if item >= 2 and t < first_seen[item]:
                cat[item], duration[item], first_seen[item] = category, dur, t
        if label == 1 and (protocol == 'rolling' or t < splitter.train_end):
            heapq.heappush(pending, (available, sequence, (item, watch, dur, t)))
        sequence += 1
        if len(buffers[split]) >= shard_size:
            flush(split)
    for split in buffers:
        flush(split)
    if not all(counts.values()):
        raise ValueError(f'All splits must be nonempty; inspect timestamp boundaries: {counts}')
    for name, array in [('popularity',popularity),('activity',activity),('catalog_category',cat),
                        ('catalog_duration',duration),('catalog_first_seen',first_seen)]:
        np.save(output/f'{name}.npy', array)
    mean = dense_sum/fitted_count
    std = np.sqrt(np.maximum(dense_sq/fitted_count-mean**2, 1e-6))
    statistics = {'dense_names':DENSE_NAMES,'dense_mean':mean.tolist(),'dense_std':std.tolist(),
                  'fitted_rows':fitted_count,'fit_split':'train'}
    write_json(output/'features.json', statistics)
    meta_path = source.with_suffix(source.suffix+'.meta.json')
    metadata = json.loads(meta_path.read_text()) if meta_path.exists() else {'adapter':{'kind':'generic'}}
    if metadata.get('adapter',{}).get('kind') == 'kuairand' and availability['conversion']:
        raise ValueError('KuaiRand must not contain purchase conversion labels')
    cardinalities = {'users':db.execute('SELECT count(DISTINCT u) FROM events').fetchone()[0],
                     'items':db.execute('SELECT count(DISTINCT i) FROM events').fetchone()[0], 'exposures':total}
    db.execute('DROP TABLE events')
    db.commit()
    db.execute('VACUUM')
    db.close()
    assets = ['events.db','features.json','popularity.npy','activity.npy','catalog_category.npy',
              'catalog_duration.npy','catalog_first_seen.npy']
    manifest = {'schema_version':2,'max_len':length,'n_items':n_items,'n_users':n_users,
                'n_categories':n_categories,'cardinalities':cardinalities,'counts':counts,'shards':shards,'oov':oov,
                'train_labels':availability,'primary_label':primary,'history_protocol':protocol,
                'train_end':splitter.train_end,'valid_end':splitter.valid_end,
                'source_sha256':file_hash(source),'adapter_metadata':metadata,
                'deduplicated':source_count-total,'assets':{f:file_hash(output/f) for f in assets},
                'data_config':spec}
    manifest['fingerprint'] = fingerprint(manifest)
    write_json(output/'manifest.json', manifest)
    return manifest


def load_artifacts(root: str | Path, verify: bool = True) -> tuple[dict, dict, dict]:
    root = Path(root)
    manifest = json.loads((root/'manifest.json').read_text())
    if manifest.get('schema_version') != 2:
        raise ValueError('Expected v2 dataset')
    identity = dict(manifest)
    expected = identity.pop('fingerprint')
    if fingerprint(identity) != expected:
        raise ValueError('Manifest fingerprint mismatch')
    if verify:
        for name, digest in manifest['assets'].items():
            if file_hash(root/name) != digest:
                raise ValueError(f'Artifact changed: {name}')
    statistics = json.loads((root/'features.json').read_text())
    catalog = {name:np.load(root/f'catalog_{name}.npy', mmap_mode='r') for name in ('duration','category','first_seen')}
    return manifest, statistics, catalog
