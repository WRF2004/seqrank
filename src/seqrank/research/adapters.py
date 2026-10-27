"""Streaming adapters. Labels are never silently substituted across semantics."""
from __future__ import annotations

import csv
import hashlib
import sqlite3
import tempfile
import glob
import math
import re
from pathlib import Path
from typing import Iterator

from .config import file_hash, write_json

FIELDS = ['user_id', 'item_id', 'timestamp', 'click', 'valid_play', 'dwell', 'duration',
          'conversion', 'category', 'tab', 'policy', 'available_at']


def paths_for(patterns: list[str]) -> list[Path]:
    paths = sorted({Path(p) for pattern in patterns for p in glob.glob(pattern)})
    if not paths:
        raise FileNotFoundError(f'No input files match {patterns}; no synthetic fallback')
    return paths


def rows(path: str | Path) -> Iterator[dict]:
    path = Path(path)
    if path.suffix.lower() == '.parquet':
        import pyarrow.parquet as pq
        for batch in pq.ParquetFile(path).iter_batches(batch_size=8192):
            yield from batch.to_pylist()
    elif path.suffix.lower() == '.csv':
        with path.open(newline='', encoding='utf-8-sig') as source:
            yield from csv.DictReader(source)
    else:
        raise ValueError(f'Expected CSV or Parquet: {path}')


def number(value, default=-1.0) -> float:
    if value is None or str(value).strip().lower() in {'', 'nan', 'none', 'null'}:
        return default
    out = float(value)
    if not math.isfinite(out):
        raise ValueError(f'Non-finite numeric value: {value}')
    return out


def binary(value) -> int:
    result = number(value)
    if result not in (-1, 0, 1):
        raise ValueError(f'Expected binary label or missing, got {value}')
    return int(result)


def inspect(config: dict) -> dict:
    adapter = config['adapter']
    result = []
    for path in paths_for(adapter['sources']):
        first = next(rows(path), None)
        if first is None:
            raise ValueError(f'Empty input: {path}')
        required = {'user_id', 'video_id', 'time_ms'} if adapter['kind'] == 'kuairand' else {'user_id', 'item_id', 'timestamp'}
        missing = required - first.keys()
        if missing:
            raise ValueError(f'{path}: missing columns {sorted(missing)}')
        version = re.search(r'(27k|1k|pure)', str(path), re.I)
        declared = str(adapter.get('version','auto')).lower()
        if adapter['kind']=='kuairand' and declared not in {'auto','27k','1k','pure'}:
            raise ValueError('KuaiRand version must be auto/27k/1k/pure')
        if declared!='auto' and version and declared!=version.group(1).lower():
            raise ValueError(f'Configured version {declared} conflicts with filename {path}')
        result.append({'path': str(path), 'columns': list(first),
                       'version_hint': version.group(1).lower() if version else 'unknown',
                       'version_hint_source': 'filename; not inferred from cardinality'})
    return {'files': result, 'adapter': adapter['kind'], 'configured_version': adapter.get('version', 'auto')}


def adapt_row(row: dict, spec: dict, path: Path) -> dict | None:
    kind = spec['kind']
    if int(spec.get('feedback_delay_ms',0))<0:
        raise ValueError('feedback_delay_ms must be nonnegative')
    if kind not in {'kuairand', 'generic'}:
        raise ValueError(f'Unknown adapter: {kind}')
    if kind == 'kuairand':
        required = {'user_id', 'video_id', 'time_ms'}
        if not required <= row.keys():
            raise ValueError(f'{path}: missing {required - row.keys()}')
        tab = str(row.get('tab', 'unknown'))
        if spec.get('tabs') and tab not in [str(x) for x in spec['tabs']]:
            return None
        policy = 'random' if binary(row.get('is_rand')) == 1 or 'log_random' in path.name else 'standard'
        if spec.get('policy', 'standard') != 'all' and policy != spec.get('policy', 'standard'):
            return None
        dwell, duration = number(row.get('play_time_ms')), number(row.get('duration_ms'))
        dwell = dwell / 1000 if dwell >= 0 else -1
        duration = duration / 1000 if duration > 0 else -1
        # Official docs do not map every tab ID to a UI. Require explicit mapping for CTR.
        semantic = spec.get('tab_semantics', {}).get(tab, spec.get('tab_semantics', {}).get(int(tab) if tab.isdigit() else tab))
        click = binary(row.get('is_click')) if semantic == 'click' else -1
        valid = binary(row.get('is_click')) if semantic == 'valid_play' else -1
        if valid < 0 and dwell >= 0 and duration > 0:
            valid = int(dwell >= duration if duration <= 7 else dwell > 7)
        primary = spec.get('primary_label', 'valid_play')
        if primary == 'click' and semantic != 'click':
            raise ValueError(f'{path}: tab={tab} needs verified tab_semantics=click or a filtered tab list; no guessed CTR')
        event = dict(user_id=str(row['user_id']), item_id=str(row['video_id']), timestamp=int(row['time_ms']),
                     click=click, valid_play=valid, dwell=dwell, duration=duration, conversion=-1,
                     category=str(row.get(spec.get('category_column', 'category'), 'unknown') or 'unknown'),
                     tab=tab, policy=policy, available_at=-1)
    else:
        unit = spec.get('timestamp_unit', 'ms')
        if unit not in {'s', 'ms', 'us'}:
            raise ValueError('timestamp_unit must be s, ms or us')
        scale = {'s': 1000, 'ms': 1, 'us': .001}[unit]
        event = dict(user_id=str(row['user_id']), item_id=str(row['item_id']),
                     timestamp=int(number(row['timestamp']) * scale),
                     click=binary(row.get('click')), valid_play=binary(row.get('valid_play')),
                     conversion=binary(row.get('conversion')), dwell=number(row.get('dwell')),
                     duration=number(row.get('duration')), category=str(row.get('category') or 'unknown'),
                     tab=str(row.get('tab') or 'unknown'), policy=str(row.get('policy') or 'unknown'),
                     available_at=int(number(row.get('available_at')) * scale) if number(row.get('available_at')) >= 0 else -1)
    if not event['user_id'] or not event['item_id'] or event['timestamp'] < 0:
        raise ValueError('Nonempty IDs and nonnegative event time required')
    if event['conversion'] == 1 and event['click'] != 1:
        raise ValueError('Observed conversion requires observed click=1')
    for key in ('duration', 'dwell'):
        if event[key] < 0 and event[key] != -1:
            raise ValueError(f'{key} must be nonnegative or -1')
    original_watch = max(event['dwell'], 0)
    cap = spec.get('max_watch_seconds', 86400)
    if event['dwell'] > cap:
        policy = spec.get('outlier_policy', 'error')
        if policy == 'error':
            raise ValueError(f'Watch time exceeds max_watch_seconds={cap}')
        if policy == 'drop':
            return None
        if policy != 'clip':
            raise ValueError('outlier must be error, drop or clip')
        event['dwell'] = float(cap)
    loop = spec.get('loop_policy', 'keep')
    if loop not in {'keep', 'clip', 'drop'}:
        raise ValueError('loop_policy must be keep, clip or drop')
    if event['duration'] > 0 and event['dwell'] > event['duration']:
        if loop == 'drop':
            return None
        if loop == 'clip':
            event['dwell'] = event['duration']
    if event['available_at'] < 0:
        # Conservative availability: do not consume watch feedback while playback is unfinished.
        event['available_at'] = event['timestamp'] + max(1, int(original_watch * 1000)) + int(spec.get('feedback_delay_ms', 0))
    if event['available_at'] <= event['timestamp']:
        raise ValueError('available_at must be strictly after timestamp')
    return event


def convert(config: dict) -> dict:
    spec = config['adapter']
    output = Path(spec['output'])
    if output.exists() or output.with_suffix(output.suffix + '.meta.json').exists():
        raise FileExistsError(f'Refusing to overwrite {output}')
    schema = inspect(config)
    output.parent.mkdir(parents=True, exist_ok=True)
    fraction = float(spec.get('user_sample_fraction', 1))
    if not 0 < fraction <= 1:
        raise ValueError('user_sample_fraction must be in (0,1]')
    # Only immutable content attributes are supported; popularity/engagement snapshots are forbidden.
    content = spec.get('content')
    temporary = tempfile.TemporaryDirectory(prefix='seqrank-content-')
    db = sqlite3.connect(str(Path(temporary.name)/'content.db'))
    db.execute('CREATE TABLE content(id TEXT PRIMARY KEY,duration REAL,category TEXT)')
    if content:
        if not content.get('assert_immutable', False):
            raise ValueError('Content join requires assert_immutable=true; dynamic snapshots need an upstream as-of join')
        for path in paths_for(content['sources']):
            for row in rows(path):
                key = content.get('id_column','video_id')
                if key not in row:
                    raise ValueError(f'Content file lacks {key}: {path}')
                duration = number(row.get(content.get('duration_column','video_duration')))
                category = str(row.get(content.get('category_column','tag')) or 'unknown')
                value = (str(row[key]),duration,category)
                old = db.execute('SELECT duration,category FROM content WHERE id=?',(value[0],)).fetchone()
                if old and old!=value[1:]:
                    raise ValueError('Conflicting content snapshots; provide immutable or as-of features')
                db.execute('INSERT OR IGNORE INTO content VALUES(?,?,?)',value)
        db.commit()
    count = dropped = 0
    with output.open('x', newline='') as target:
        writer = csv.DictWriter(target, fieldnames=FIELDS)
        writer.writeheader()
        for path in paths_for(spec['sources']):
            for line, row in enumerate(rows(path), 2):
                try:
                    user = str(row.get('user_id',''))
                    draw = int.from_bytes(hashlib.sha256((str(spec.get('sample_seed',42))+':'+user).encode()).digest()[:8],'big')/2**64
                    if draw >= fraction:
                        dropped += 1
                        continue
                    if content:
                        key = 'video_id' if spec['kind']=='kuairand' else 'item_id'
                        found = db.execute('SELECT duration,category FROM content WHERE id=?',(str(row[key]),)).fetchone()
                        if found:
                            row = dict(row)
                            column = 'duration_ms' if spec['kind']=='kuairand' else 'duration'
                            if number(row.get(column))<=0 and found[0]>0:
                                unit = content.get('duration_unit','ms')
                                if unit not in {'s','ms'}:
                                    raise ValueError('content.duration_unit must be s/ms')
                                seconds = found[0]/1000 if unit=='ms' else found[0]
                                row[column] = seconds*1000 if spec['kind']=='kuairand' else seconds
                            row[spec.get('category_column','category')] = found[1]
                    event = adapt_row(row, spec, path)
                except (ValueError, KeyError) as exc:
                    raise ValueError(f'{path}:{line}: {exc}') from exc
                if event is None:
                    dropped += 1
                else:
                    writer.writerow(event)
                    count += 1
    db.close()
    temporary.cleanup()
    if not count:
        raise ValueError('No usable rows after filtering')
    metadata = {'adapter': spec, 'schema': schema, 'rows': count, 'dropped': dropped,
                'source_hashes': {str(p): file_hash(p) for p in paths_for(spec['sources'])},
                'normalized_sha256': file_hash(output), 'timestamp_unit': 'ms',
                'conversion_available': spec['kind'] != 'kuairand'}
    if content:
        metadata['content_hashes'] = {str(p):file_hash(p) for p in paths_for(content['sources'])}
    write_json(output.with_suffix(output.suffix + '.meta.json'), metadata)
    return metadata
