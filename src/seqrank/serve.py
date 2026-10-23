import json
import sqlite3
import time
from pathlib import Path

import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, model_validator

from .engine import load_model, sha256
from .retrieval import ANNIndex, exact_topk


class Request(BaseModel):
    user_id: str = Field(max_length=256)
    history: list[str] = Field(default_factory=list, max_length=1000)
    k: int = Field(default=10, ge=1, le=100)
    candidates: int = Field(default=100, ge=1, le=1000)
    exclude_seen: bool = False

    @model_validator(mode="after")
    def validate_candidates(self):
        if self.candidates < self.k or any(len(x) > 256 for x in self.history):
            raise ValueError("candidates must be >= k; IDs must be <=256 characters")
        return self


def create_app(data, retrieval_path, ranker_path, ann_path=None, device="cpu"):
    retrieval, rc = load_model(retrieval_path, device)
    ranker, mc = load_model(ranker_path, device)
    manifest = json.loads((Path(data) / "manifest.json").read_text())
    if manifest != rc["manifest"] or manifest != mc["manifest"]:
        raise ValueError("Models/data mismatch")
    retrieval_hash, rank_hash = sha256(retrieval_path), sha256(ranker_path)
    ann = ANNIndex(ann_path, retrieval_hash) if ann_path else None
    app = FastAPI(title="SeqRank", version="0.1.0")
    database = (Path(data) / "events.db").resolve().as_uri() + "?mode=ro"

    @app.get("/health")
    def health():
        return {
            "status": "ready",
            "retrieval_version": retrieval_hash,
            "ranker_version": rank_hash,
            "backend": "hnsw" if ann else "exact",
        }

    @app.post("/recommend")
    def recommend(request: Request):
        start = time.perf_counter()
        # Request-scoped read-only connections; avoid cross-thread SQLite connection reuse.
        with sqlite3.connect(database, uri=True) as conn:

            def lookup(table, key):
                result = conn.execute(f"SELECT id FROM {table} WHERE raw=?", (key,)).fetchone()
                return result[0] if result else 1

            uid = lookup("user_vocab", request.user_id)
            mapped = [lookup("item_vocab", raw) for raw in request.history]
            history = mapped[-manifest["max_len"] :]
            history += [0] * (manifest["max_len"] - len(history))
            h = torch.tensor([history], device=device)
            with torch.inference_mode():
                if ann:
                    # Over-fetch by seen-item count, then filter; actual result may be shorter.
                    fetch = request.candidates + (len(set(mapped)) if request.exclude_seen else 0)
                    _, found = ann.search(retrieval(h).cpu().numpy(), fetch)
                    ids = [int(i) for i in found[0] if i >= 2]
                else:
                    _, found = exact_topk(
                        retrieval,
                        h,
                        request.candidates + (len(set(mapped)) if request.exclude_seen else 0),
                        rc["config"]["chunk_size"],
                    )
                    ids = found[0].tolist()
                if request.exclude_seen:
                    seen = set(mapped)
                    ids = [i for i in ids if i not in seen]
                ids = ids[: request.candidates]
                if not ids:
                    return {
                        "items": [],
                        "cold_user": uid == 1,
                        "latency_ms": (time.perf_counter() - start) * 1000,
                    }
                item_tensor = torch.tensor(ids, device=device)
                output = ranker(torch.full_like(item_tensor, uid), h.expand(len(ids), -1), item_tensor)
                score = (
                    output["ctr"]
                    + float(mc["config"]["conversion_weight"] > 0) * output["ctcvr"]
                    + 0.01 * float(mc["config"]["dwell_weight"] > 0) * output["log_dwell"]
                )
                order = score.argsort(descending=True)[: request.k].tolist()
                items = []
                for pos in order:
                    raw = conn.execute("SELECT raw FROM item_vocab WHERE id=?", (ids[pos],)).fetchone()
                    if not raw:
                        raise HTTPException(503, "Catalog mapping missing")
                    items.append(
                        {
                            "item_id": raw[0],
                            "score": score[pos].item(),
                            "p_click": output["ctr"][pos].item(),
                            "p_conversion": output["ctcvr"][pos].item()
                            if mc["config"]["conversion_weight"] > 0
                            else None,
                            "expected_log_dwell": output["log_dwell"][pos].item(),
                        }
                    )
        return {
            "items": items,
            "cold_user": uid == 1,
            "retrieval_version": retrieval_hash,
            "ranker_version": rank_hash,
            "latency_ms": (time.perf_counter() - start) * 1000,
        }

    return app
