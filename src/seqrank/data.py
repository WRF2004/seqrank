"""Disk-backed ingestion: SQLite external sort -> fixed-width NPY shards.

Input must be mature impression labels. Histories contain strictly earlier clicks.
ID 0 is padding, 1 is OOV; vocabularies are fitted only before train_end.
"""

import bisect
import csv
import hashlib
import json
import math
import random
import sqlite3
from collections import deque
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


def synthetic(path, users=200, items=200, steps=50, seed=42):
    if users < 1 or items < 20 or steps < 10:
        raise ValueError("Require users>=1, items>=20, steps>=10")
    rng = random.Random(seed)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["user_id", "item_id", "timestamp", "click", "conversion", "dwell"])
        for u in range(users):
            interest = u % 10
            for t in range(steps):
                item = rng.randrange(items)
                click = int(rng.random() < (0.8 if item % 10 == interest else 0.12))
                conversion = int(click and rng.random() < (0.3 if item % 10 == interest else 0.03))
                dwell = round(rng.expovariate(1 / (30 if item % 10 == interest else 5)), 2) if click else 0
                writer.writerow([u, item, t * users + u, click, conversion, dwell])


def dtype_for(length):
    return np.dtype(
        [
            ("user", "<i8"),
            ("item", "<i8"),
            ("timestamp", "<i8"),
            ("history", "<i8", (length,)),
            ("click", "<f4"),
            ("conversion", "<f4"),
            ("dwell", "<f4"),
        ]
    )


def prepare(source, out, train_end, valid_end, max_len=50, shard_size=100000):
    if train_end >= valid_end or max_len < 1 or shard_size < 1:
        raise ValueError("Invalid split, sequence length or shard size")
    out = Path(out)
    if out.exists() and any(out.iterdir()):
        raise ValueError("Output directory must be empty; prevent mixing dataset versions")
    out.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(out / "events.db")
    conn.execute("PRAGMA temp_store=FILE")
    conn.execute("CREATE TABLE events(u TEXT,i TEXT,t INTEGER,c REAL,v REAL,d REAL)")
    digest = hashlib.sha256()
    with open(source, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(block)
    with open(source, newline="") as f:
        rows = csv.DictReader(f)
        if not {"user_id", "item_id", "timestamp", "click"} <= set(rows.fieldnames or []):
            raise ValueError("Missing required CSV columns")
        batch = []
        for row in rows:
            c = float(row["click"])
            v = float(row.get("conversion") or -1)
            d = float(row.get("dwell") or -1)
            if c not in (0, 1) or v not in (-1, 0, 1) or v > c:
                raise ValueError("click/conversion must be binary, conversion<=click; -1 means missing")
            if not math.isfinite(d) or (d < 0 and d != -1):
                raise ValueError("dwell must be finite nonnegative seconds or -1")
            batch.append((row["user_id"], row["item_id"], int(row["timestamp"]), c, v, d))
            if len(batch) >= 10000:
                conn.executemany("INSERT INTO events VALUES(?,?,?,?,?,?)", batch)
                batch.clear()
        conn.executemany("INSERT INTO events VALUES(?,?,?,?,?,?)", batch)
    conn.execute("CREATE INDEX user_time ON events(u,t)")
    conn.execute("CREATE TABLE item_vocab(raw TEXT PRIMARY KEY,id INTEGER UNIQUE)")
    conn.execute("CREATE TABLE user_vocab(raw TEXT PRIMARY KEY,id INTEGER UNIQUE)")
    for kind, col in [("item", "i"), ("user", "u")]:
        cursor = conn.execute(f"SELECT DISTINCT {col} FROM events WHERE t < ? ORDER BY {col}", (train_end,))
        conn.executemany(
            f"INSERT INTO {kind}_vocab VALUES(?,?)", ((r[0], j + 2) for j, r in enumerate(cursor))
        )
    conn.commit()
    n_items = conn.execute("SELECT count(*) FROM item_vocab").fetchone()[0] + 2
    n_users = conn.execute("SELECT count(*) FROM user_vocab").fetchone()[0] + 2
    if n_items < 4:
        raise ValueError("Need at least two training catalog items")
    buffers = {s: [] for s in ("train", "valid", "test")}
    shards = {s: [] for s in buffers}
    counts = {s: 0 for s in buffers}
    popularity = np.zeros(n_items, dtype=np.int64)

    def flush(split):
        if not buffers[split]:
            return
        filename = f"{split}-{len(shards[split]):05d}.npy"
        np.save(out / filename, np.array(buffers[split], dtype=dtype_for(max_len)))
        shards[split].append({"file": filename, "rows": len(buffers[split])})
        buffers[split].clear()

    query = """SELECT e.u,COALESCE(uv.id,1),COALESCE(iv.id,1),e.t,e.c,e.v,e.d
               FROM events e LEFT JOIN user_vocab uv ON uv.raw=e.u
               LEFT JOIN item_vocab iv ON iv.raw=e.i ORDER BY e.u,e.t,e.rowid"""
    history, pending = deque(maxlen=max_len), []
    previous_user, previous_t = None, None
    for raw_user, user, item, t, c, v, d in conn.execute(query):
        if raw_user != previous_user:
            history.clear()
            pending.clear()
        elif t != previous_t:
            history.extend(pending)
            pending.clear()
        # Same-timestamp interactions never see each other.
        previous_user, previous_t = raw_user, t
        split = "train" if t < train_end else "valid" if t < valid_end else "test"
        h = list(history)
        h += [0] * (max_len - len(h))
        buffers[split].append((user, item, t, h, c, v, d))
        counts[split] += 1
        if c:
            pending.append(item)
            if split == "train":
                popularity[item] += 1
        if len(buffers[split]) >= shard_size:
            flush(split)
    for split in buffers:
        flush(split)
    np.save(out / "popularity.npy", popularity)
    manifest = {
        "schema_version": 1,
        "source_sha256": digest.hexdigest(),
        "max_len": max_len,
        "train_end": train_end,
        "valid_end": valid_end,
        "n_items": n_items,
        "n_users": n_users,
        "counts": counts,
        "shards": shards,
        "protocol": "global temporal split; rolling strictly earlier click history",
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    # Persist only mappings, not raw impressions, after materialization.
    conn.execute("DROP TABLE events")
    conn.commit()
    conn.execute("VACUUM")
    conn.close()
    return manifest


class ShardedDataset(Dataset):
    """Lazy mmap access. Each worker opens its own shard handles."""

    def __init__(self, root, split):
        self.root = Path(root)
        self.manifest = json.loads((self.root / "manifest.json").read_text())
        self.shards = self.manifest["shards"][split]
        self.ends = np.cumsum([x["rows"] for x in self.shards]).tolist()
        self.cache = {}

    def __len__(self):
        return self.ends[-1] if self.ends else 0

    def __getitem__(self, idx):
        if idx < 0 or idx >= len(self):
            raise IndexError(idx)
        shard = bisect.bisect_right(self.ends, idx)
        if shard not in self.cache:
            if len(self.cache) >= 8:
                self.cache.clear()
            self.cache[shard] = np.load(self.root / self.shards[shard]["file"], mmap_mode="r")
        row = self.cache[shard][idx - (self.ends[shard - 1] if shard else 0)]
        return {key: torch.as_tensor(np.array(row[key])) for key in row.dtype.names}

    def __getstate__(self):
        state = self.__dict__.copy()
        state["cache"] = {}
        return state
