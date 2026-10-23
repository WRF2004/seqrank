"""Bounded-score-memory exact top-k and optional FAISS HNSW inner-product index."""

import json
from pathlib import Path

import numpy as np
import torch


@torch.inference_mode()
def exact_topk(model, history, k=100, chunk_size=8192, exclude_seen=False):
    if k < 1 or chunk_size < 1:
        raise ValueError("k and chunk_size must be positive")
    q = model(history)
    k = min(k, model.item.num_embeddings - 2)
    scores = q.new_empty((q.size(0), 0))
    ids = torch.empty((q.size(0), 0), dtype=torch.long, device=q.device)
    for start in range(2, model.item.num_embeddings, chunk_size):
        stop = min(start + chunk_size, model.item.num_embeddings)
        s = q @ model.item.weight[start:stop].T
        if exclude_seen:
            # Avoid B*L*chunk allocation.
            rows = torch.arange(q.size(0), device=q.device)[:, None].expand_as(history)
            valid = (history >= start) & (history < stop)
            s[rows[valid], history[valid] - start] = -torch.inf
        i = torch.arange(start, stop, device=q.device).expand(q.size(0), -1)
        merged_scores, merged_ids = torch.cat([scores, s], 1), torch.cat([ids, i], 1)
        scores, positions = merged_scores.topk(min(k, merged_scores.size(1)), dim=1)
        ids = merged_ids.gather(1, positions)
    return scores, ids


@torch.inference_mode()
def build_ann(model, output, checkpoint_sha256, m=32, ef_construction=200, batch_size=10000):
    import faiss

    index = faiss.IndexHNSWFlat(model.item.embedding_dim, m, faiss.METRIC_INNER_PRODUCT)
    index.hnsw.efConstruction = ef_construction
    for start in range(2, model.item.num_embeddings, batch_size):
        index.add(
            np.ascontiguousarray(model.item.weight[start : start + batch_size].cpu().numpy(), dtype="float32")
        )
    output = Path(output)
    faiss.write_index(index, str(output))
    output.with_suffix(output.suffix + ".json").write_text(
        json.dumps(
            {
                "checkpoint_sha256": checkpoint_sha256,
                "id_offset": 2,
                "ntotal": index.ntotal,
                "metric": "inner_product",
                "m": m,
                "ef_construction": ef_construction,
            },
            indent=2,
        )
    )


class ANNIndex:
    def __init__(self, path, checkpoint_sha256, ef_search=128):
        import faiss

        path = Path(path)
        metadata = json.loads(path.with_suffix(path.suffix + ".json").read_text())
        if metadata["checkpoint_sha256"] != checkpoint_sha256:
            raise ValueError("ANN index and retrieval checkpoint do not match")
        self.index = faiss.read_index(str(path))
        self.index.hnsw.efSearch = ef_search

    def search(self, queries, k):
        k = min(k, self.index.ntotal)
        scores, ids = self.index.search(np.ascontiguousarray(queries, dtype="float32"), k)
        return scores, ids + 2
