import argparse
import json
from pathlib import Path

import torch
import yaml

from .data import prepare, synthetic
from .engine import evaluate, load_model, sha256, train


def main():
    import sys
    from .research.cli import COMMANDS, main as research_main
    if len(sys.argv)>1 and sys.argv[1] in COMMANDS:
        return research_main(sys.argv[1:])
    parser = argparse.ArgumentParser(description="SeqRank two-stage recommendation")
    parser.add_argument("--threads", type=int, default=4, help="PyTorch CPU threads per process")
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("demo-data")
    p.add_argument("--out", default="data/events.csv")
    p.add_argument("--users", type=int, default=200)
    p.add_argument("--items", type=int, default=200)
    p.add_argument("--steps", type=int, default=50)
    p.add_argument("--seed", type=int, default=42)
    p = sub.add_parser("prepare")
    p.add_argument("--source", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--train-end", required=True, type=int)
    p.add_argument("--valid-end", required=True, type=int)
    p.add_argument("--max-len", type=int, default=20)
    p.add_argument("--shard-size", type=int, default=100000)
    p = sub.add_parser("train")
    p.add_argument("--stage", choices=["retrieval", "ranker"], required=True)
    p.add_argument("--data", required=True)
    p.add_argument("--out", default="runs/demo")
    p.add_argument("--config", default="configs/demo.yaml")
    p.add_argument("--device")
    p.add_argument("--resume")
    p = sub.add_parser("evaluate")
    p.add_argument("--data", required=True)
    p.add_argument("--retrieval", required=True)
    p.add_argument("--ranker", required=True)
    p.add_argument("--split", choices=["valid", "test"], default="test")
    p.add_argument("--out", required=True)
    p.add_argument("--device", default="cpu")
    p.add_argument("--k", type=int, default=10)
    p.add_argument("--candidates", type=int, default=50)
    p = sub.add_parser("build-index")
    p.add_argument("--retrieval", required=True)
    p.add_argument("--out", required=True)
    p = sub.add_parser("serve")
    p.add_argument("--data", required=True)
    p.add_argument("--retrieval", required=True)
    p.add_argument("--ranker", required=True)
    p.add_argument("--ann")
    p.add_argument("--device", default="cpu")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    if args.command == "demo-data":
        synthetic(args.out, args.users, args.items, args.steps, args.seed)
    elif args.command == "prepare":
        print(
            json.dumps(
                prepare(args.source, args.out, args.train_end, args.valid_end, args.max_len, args.shard_size),
                indent=2,
            )
        )
    elif args.command == "train":
        config = yaml.safe_load(Path(args.config).read_text())
        train(args.data, args.out, config, args.stage, args.device, args.resume)
    elif args.command == "evaluate":
        report = evaluate(
            args.data, args.retrieval, args.ranker, args.split, args.device, args.k, args.candidates
        )
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(report, indent=2, allow_nan=False))
        print(json.dumps(report, indent=2))
    elif args.command == "build-index":
        from .retrieval import build_ann

        model, _ = load_model(args.retrieval)
        build_ann(model, args.out, sha256(args.retrieval))
    elif args.command == "serve":
        import uvicorn

        from .serve import create_app

        uvicorn.run(
            create_app(args.data, args.retrieval, args.ranker, args.ann, args.device),
            host=args.host,
            port=args.port,
        )


if __name__ == "__main__":
    main()
