"""Measured in-process end-to-end request latency. Not an HTTP/QPS load test."""

import argparse
import json
import time

import numpy as np
import torch
from fastapi.testclient import TestClient

from seqrank.serve import create_app

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    p.add_argument("--run", required=True)
    p.add_argument("--ann")
    p.add_argument("--requests", type=int, default=100)
    args = p.parse_args()
    torch.set_num_threads(1)
    client = TestClient(
        create_app(args.data, f"{args.run}/retrieval-best.pt", f"{args.run}/ranker-best.pt", args.ann)
    )
    request = {"user_id": "0", "history": ["0", "10", "20"], "k": 10, "candidates": 50}
    latencies = []
    for i in range(args.requests + 10):
        start = time.perf_counter()
        response = client.post("/recommend", json=request)
        response.raise_for_status()
        if i >= 10:
            latencies.append((time.perf_counter() - start) * 1000)
    print(
        json.dumps(
            {
                "scope": "in-process TestClient; CPU single request; 10 warmups",
                "requests": args.requests,
                "p50_ms": float(np.percentile(latencies, 50)),
                "p95_ms": float(np.percentile(latencies, 95)),
                "p99_ms": float(np.percentile(latencies, 99)),
            },
            indent=2,
        )
    )
