import hashlib
import json
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler, Sampler

from .data import ShardedDataset
from .metrics import BinaryMetrics
from .models import Ranker, SequenceEncoder, rank_loss, sampled_loss
from .retrieval import exact_topk


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def to_device(batch, device):
    return {k: v.to(device, non_blocking=True) for k, v in batch.items()}


def model_for(stage, manifest, config):
    return (SequenceEncoder if stage == "retrieval" else Ranker)(
        n_items=manifest["n_items"], n_users=manifest["n_users"], **config
    )


def load_model(path, device="cpu"):
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    model = model_for(checkpoint["stage"], checkpoint["manifest"], checkpoint["config"]).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    return model, checkpoint


class EvalSampler(Sampler):
    """No padding or duplicated observations in distributed validation."""

    def __init__(self, size, rank, world):
        self.size, self.rank, self.world = size, rank, world

    def __iter__(self):
        return iter(range(self.rank, self.size, self.world))

    def __len__(self):
        return len(range(self.rank, self.size, self.world))


def objective(model, stage, batch, config):
    if stage == "retrieval":
        return sampled_loss(model, batch, config["negatives"])
    output = model(batch["user"], batch["history"], batch["item"])
    return rank_loss(output, batch, config["conversion_weight"], config["dwell_weight"])


def train(data, output, config, stage, device=None, resume=None):
    world, rank, local_rank = [
        int(os.getenv(k, d)) for k, d in [("WORLD_SIZE", "1"), ("RANK", "0"), ("LOCAL_RANK", "0")]
    ]
    device = torch.device(device or (f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu"))
    if device.type == "cuda":
        torch.cuda.set_device(device)
    if world > 1:
        dist.init_process_group("nccl" if device.type == "cuda" else "gloo")
    seed = config["seed"]
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    dataset = ShardedDataset(data, "train")
    valid = ShardedDataset(data, "valid")
    if not len(dataset) or not len(valid):
        raise ValueError("Training and validation splits must both be nonempty")
    if config["max_len"] != dataset.manifest["max_len"]:
        raise ValueError("Config max_len differs from prepared data")
    sampler = DistributedSampler(dataset, world, rank, seed=seed) if world > 1 else None
    loader = DataLoader(
        dataset,
        batch_size=config["batch_size"],
        sampler=sampler,
        shuffle=sampler is None,
        num_workers=config["workers"],
        pin_memory=device.type == "cuda",
        multiprocessing_context="spawn" if config["workers"] else None,
    )
    validation = DataLoader(
        valid, batch_size=config["batch_size"], sampler=EvalSampler(len(valid), rank, world), num_workers=0
    )
    model = model_for(stage, dataset.manifest, config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["lr"], weight_decay=1e-4)
    amp = config["amp"] and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    start, best = 0, float("inf")
    if resume:
        state = torch.load(resume, map_location=device, weights_only=True)
        if state["stage"] != stage or state["manifest"] != dataset.manifest or state["config"] != config:
            raise ValueError("Resume stage/config/data mismatch; use same config and total epochs")
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scaler.load_state_dict(state["scaler"])
        start, best = state["epoch"] + 1, state["best"]
    raw = model
    if world > 1:
        model = DDP(model, device_ids=[device.index] if device.type == "cuda" else None)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    for epoch in range(start, config["epochs"]):
        # Reproducible epoch-boundary resumption; dropout/negative sampling differ by rank.
        torch.manual_seed(seed + epoch * world + rank)
        np.random.seed(seed + epoch * world + rank)
        random.seed(seed + epoch * world + rank)
        if sampler:
            sampler.set_epoch(epoch)
        model.train()
        totals = torch.zeros(2, device=device, dtype=torch.float64)
        started = time.perf_counter()
        for batch in loader:
            batch = to_device(batch, device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=amp, dtype=torch.float16):
                loss = objective(model, stage, batch, config)
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite training loss")
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(optimizer)
            scaler.update()
            totals[0] += loss.detach().double() * batch["item"].size(0)
            totals[1] += batch["item"].size(0)
        raw.eval()
        val = torch.zeros(2, device=device, dtype=torch.float64)
        # Fixed validation negatives each epoch; raw avoids uneven DDP forward collectives.
        with torch.random.fork_rng(devices=[device.index] if device.type == "cuda" else []):
            torch.manual_seed(seed + 10000 + rank)
            with torch.inference_mode():
                for batch in validation:
                    batch = to_device(batch, device)
                    loss = objective(raw, stage, batch, config)
                    val[0] += loss.double() * batch["item"].size(0)
                    val[1] += batch["item"].size(0)
        if world > 1:
            dist.all_reduce(totals)
            dist.all_reduce(val)
        val_loss = (val[0] / val[1]).item()
        improved = val_loss < best
        best = min(best, val_loss)
        if rank == 0:
            record = {
                "epoch": epoch,
                "stage": stage,
                "train_loss": (totals[0] / totals[1]).item(),
                "valid_loss": val_loss,
                "seconds": time.perf_counter() - started,
                "world_size": world,
                "train_rows_including_sampler_padding": int(totals[1].item()),
            }
            print(json.dumps(record), flush=True)
            with (output / f"{stage}.jsonl").open("a") as f:
                f.write(json.dumps(record) + "\n")
            checkpoint = {
                "model": raw.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scaler": scaler.state_dict(),
                "config": config,
                "manifest": dataset.manifest,
                "stage": stage,
                "epoch": epoch,
                "best": best,
                "torch_version": str(torch.__version__),
            }
            for name in [f"{stage}-last.pt"] + ([f"{stage}-best.pt"] if improved else []):
                temp = output / (name + ".tmp")
                torch.save(checkpoint, temp)
                temp.replace(output / name)
        if world > 1:
            dist.barrier()
    if world > 1:
        dist.destroy_process_group()


@torch.inference_mode()
def evaluate(data, retrieval_path, ranker_path, split="test", device="cpu", k=10, candidates=50):
    if not 1 <= k <= candidates:
        raise ValueError("Require 1 <= k <= candidates")
    retrieval, rc = load_model(retrieval_path, device)
    ranker, mc = load_model(ranker_path, device)
    dataset = ShardedDataset(data, split)
    if not len(dataset):
        raise ValueError("Evaluation split is empty")
    if rc["manifest"] != mc["manifest"] or rc["manifest"] != dataset.manifest:
        raise ValueError("Models/data version mismatch")
    ctr, joint = BinaryMetrics(), BinaryMetrics()
    hit = ndcg = candidate_hit = pop_hit = rerank_hit = rerank_ndcg = 0.0
    positive_count = known_count = dwell_count = 0
    dwell_error = 0.0
    popularity = np.load(Path(data) / "popularity.npy")
    popular_ids = np.argsort(-popularity[2:])[:k] + 2
    current_user, user_metric, gauc_sum, gauc_rows = None, BinaryMetrics(), 0.0, 0
    for batch in DataLoader(dataset, batch_size=rc["config"]["batch_size"]):
        batch = to_device(batch, device)
        output = ranker(batch["user"], batch["history"], batch["item"])
        y, p = batch["click"].cpu().numpy(), output["ctr"].cpu().numpy()
        ctr.update(y, p)
        joint.update(batch["conversion"].cpu().numpy(), output["ctcvr"].cpu().numpy())
        # Dataset is user-sorted. OOV users are excluded from GAUC since distinct raw IDs collapse.
        for u, label, score in zip(batch["user"].tolist(), y, p):
            if u < 2:
                continue
            if u != current_user:
                result = user_metric.result()
                if result["auc_hist4096"] is not None:
                    gauc_sum += result["auc_hist4096"] * result["count"]
                    gauc_rows += result["count"]
                current_user, user_metric = u, BinaryMetrics()
            user_metric.update([label], [score])
        mask = (batch["click"] == 1) & (batch["dwell"] >= 0)
        dwell_error += (output["log_dwell"][mask] - batch["dwell"][mask].log1p()).abs().sum().item()
        dwell_count += mask.sum().item()
        positives = batch["click"] == 1
        positive_count += positives.sum().item()
        known_count += ((batch["item"] >= 2) & positives).sum().item()
        if not positives.any():
            continue
        h, u, target = batch["history"][positives], batch["user"][positives], batch["item"][positives]
        _, ids = exact_topk(retrieval, h, candidates, rc["config"]["chunk_size"])
        match = ids == target[:, None]
        candidate_hit += match.any(1).sum().item()
        hit += match[:, :k].any(1).sum().item()
        discount = torch.log2(torch.arange(ids.size(1), device=h.device) + 2)
        ndcg += (match[:, :k] / discount[:k]).sum().item()
        b, c = ids.shape
        ranked = ranker(
            u[:, None].expand(-1, c).reshape(-1),
            h[:, None, :].expand(-1, c, -1).reshape(b * c, -1),
            ids.reshape(-1),
        )
        # Explicit default utility; offline end-to-end relevance remains click-based.
        utility = (
            ranked["ctr"]
            + float(mc["config"]["conversion_weight"] > 0) * ranked["ctcvr"]
            + 0.01 * float(mc["config"]["dwell_weight"] > 0) * ranked["log_dwell"]
        )
        order = utility.view(b, c).argsort(dim=1, descending=True)
        reranked = match.gather(1, order)[:, :k]
        rerank_hit += reranked.any(1).sum().item()
        rerank_ndcg += (reranked / discount[:k]).sum().item()
        pop_hit += np.isin(target.cpu().numpy(), popular_ids).sum()
    result = user_metric.result()
    if result["auc_hist4096"] is not None:
        gauc_sum += result["auc_hist4096"] * result["count"]
        gauc_rows += result["count"]
    denom = positive_count or 1
    return {
        "split": split,
        "source_sha256": dataset.manifest["source_sha256"],
        "retrieval_checkpoint_sha256": sha256(retrieval_path),
        "ranker_checkpoint_sha256": sha256(ranker_path),
        "k": k,
        "candidates": candidates,
        "evaluation": "full training catalog; repeated items allowed; OOV targets count as misses",
        "ctr": ctr.result(),
        "ctcvr": joint.result(),
        "gauc_hist4096": gauc_sum / gauc_rows if gauc_rows else None,
        "gauc_eligible_rows": gauc_rows,
        "clicked_impressions": positive_count,
        "known_target_fraction": known_count / denom,
        "retrieval_hit_at_k": hit / denom,
        "retrieval_ndcg_at_k": ndcg / denom,
        "candidate_recall": candidate_hit / denom,
        "rerank_hit_at_k": rerank_hit / denom,
        "rerank_ndcg_at_k": rerank_ndcg / denom,
        "popularity_hit_at_k": float(pop_hit / denom),
        "clicked_log_dwell_mae": dwell_error / dwell_count if dwell_count else None,
    }
