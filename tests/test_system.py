import csv
import json
import socket

import numpy as np
import pytest
import torch
from fastapi.testclient import TestClient
from sklearn.metrics import roc_auc_score

from seqrank.data import ShardedDataset, prepare, synthetic
from seqrank.engine import EvalSampler, evaluate, load_model, train
from seqrank.metrics import BinaryMetrics
from seqrank.models import Ranker, SequenceEncoder, rank_loss, sampled_loss
from seqrank.retrieval import ANNIndex, build_ann, exact_topk
from seqrank.serve import create_app

torch.set_num_threads(1)


@pytest.fixture
def config():
    return {
        "seed": 42,
        "max_len": 5,
        "dim": 8,
        "heads": 2,
        "layers": 1,
        "dropout": 0.0,
        "experts": 2,
        "batch_size": 32,
        "epochs": 1,
        "lr": 0.001,
        "negatives": 4,
        "workers": 0,
        "amp": False,
        "chunk_size": 7,
        "conversion_weight": 1.0,
        "dwell_weight": 0.1,
    }


@pytest.fixture
def data(tmp_path):
    source = tmp_path / "events.csv"
    synthetic(source, users=10, items=20, steps=30)
    root = tmp_path / "prepared"
    prepare(source, root, 200, 250, max_len=5, shard_size=31)
    return root


def test_temporal_isolation_and_train_only_vocab(tmp_path):
    source = tmp_path / "events.csv"
    with source.open("w") as f:
        writer = csv.writer(f)
        writer.writerow(["user_id", "item_id", "timestamp", "click"])
        writer.writerows(
            [["u", "a", 1, 1], ["u", "b", 1, 1], ["u", "a", 2, 1], ["u", "new", 5, 1], ["u", "a", 8, 1]]
        )
    root = tmp_path / "data"
    m = prepare(source, root, 4, 7, max_len=4, shard_size=1)
    assert m["n_items"] == 4
    train_data, val, test = [ShardedDataset(root, s) for s in ["train", "valid", "test"]]
    assert train_data[0]["history"].sum() == train_data[1]["history"].sum() == 0
    assert train_data[2]["history"].tolist() == [2, 3, 0, 0]
    assert val[0]["item"] == 1
    assert val[0]["history"].tolist() == [2, 3, 2, 0]
    assert test[0]["history"].tolist() == [2, 3, 2, 1]
    assert train_data[0]["conversion"] == -1
    with pytest.raises(ValueError):
        prepare(source, root, 4, 7)


def test_invalid_conversion_rejected(tmp_path):
    p = tmp_path / "bad.csv"
    p.write_text("user_id,item_id,timestamp,click,conversion\nu,i,1,0,1\n")
    with pytest.raises(ValueError):
        prepare(p, tmp_path / "bad", 2, 3)


def test_causal_encoder_and_empty_history(config):
    m = SequenceEncoder(22, **config).eval()
    a = torch.tensor([[2, 3, 4, 0, 0], [0, 0, 0, 0, 0]])
    b = a.clone()
    b[0, 2] = 9
    assert torch.isfinite(m(a)).all()
    assert torch.allclose(m.sequence(a)[0, :3], m.sequence(b)[0, :3], atol=1e-6)
    assert not torch.allclose(m(a)[0], m(b)[0])


def test_topk_matches_dense(config):
    m = SequenceEncoder(22, **config).eval()
    h = torch.tensor([[2, 3, 0, 0, 0], [0, 0, 0, 0, 0]])
    scores, ids = exact_topk(m, h, k=10, chunk_size=3)
    dense = m(h) @ m.item.weight[2:].T
    expected = dense.topk(10)
    assert torch.equal(ids, expected.indices + 2)
    assert torch.allclose(scores, expected.values, atol=1e-5)
    _, ids = exact_topk(m, h, k=5, chunk_size=4, exclude_seen=True)
    assert 2 not in ids[0] and 3 not in ids[0]


def test_ranker_missing_labels_and_joint_constraint(config):
    model = Ranker(22, 12, **config)
    b = {
        "user": torch.tensor([1, 2]),
        "item": torch.tensor([2, 3]),
        "history": torch.zeros(2, 5, dtype=torch.long),
        "click": torch.tensor([0.0, 1.0]),
        "conversion": torch.tensor([-1.0, -1.0]),
        "dwell": torch.tensor([-1.0, -1.0]),
    }
    o = model(b["user"], b["history"], b["item"])
    assert (o["ctcvr"] <= o["ctr"]).all()
    loss = rank_loss(o, b)
    assert torch.isfinite(loss)
    loss.backward()
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)


def test_extreme_esmm_gradients():
    a = torch.tensor([100.0, -100.0], requires_grad=True)
    b = torch.tensor([100.0, -100.0], requires_grad=True)
    output = {"ctr_logit": a, "cvr_logit": b, "log_dwell": a.sigmoid()}
    labels = {
        "click": torch.tensor([0.0, 1.0]),
        "conversion": torch.tensor([0.0, 1.0]),
        "dwell": torch.tensor([-1.0, -1.0]),
    }
    loss = rank_loss(output, labels)
    loss.backward()
    assert torch.isfinite(loss) and torch.isfinite(a.grad).all() and torch.isfinite(b.grad).all()


def test_retrieval_all_negative_batch(config):
    model = SequenceEncoder(22, **config)
    batch = {
        "history": torch.zeros(2, 5, dtype=torch.long),
        "item": torch.tensor([1, 2]),
        "click": torch.zeros(2),
    }
    loss = sampled_loss(model, batch, 4)
    loss.backward()
    assert loss.item() == 0
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)


def test_metrics_ties_and_single_class():
    m = BinaryMetrics()
    y, p = [0, 1, 1, 0], [0.1, 0.5, 0.5, 0.8]
    m.update(y, p)
    assert m.result()["auc_hist4096"] == pytest.approx(roc_auc_score(y, p))
    m = BinaryMetrics()
    m.update([1, 1], [0.2, 0.3])
    assert m.result()["auc_hist4096"] is None


def test_eval_sampler_no_duplication():
    indices = [list(EvalSampler(11, r, 3)) for r in range(3)]
    assert sorted(i for part in indices for i in part) == list(range(11))


def test_mmap_multishard_and_workers(data):
    try:
        probe = socket.socket(socket.AF_UNIX)
        probe.close()
    except PermissionError:
        pytest.skip("Host prohibits Unix sockets required by PyTorch tensor IPC")
    d = ShardedDataset(data, "train")
    assert len(d) == 200
    batches = list(
        torch.utils.data.DataLoader(
            d, batch_size=17, num_workers=2, multiprocessing_context="spawn", timeout=30
        )
    )
    assert sum(len(b["item"]) for b in batches) == 200
    with pytest.raises(IndexError):
        _ = d[len(d)]


def test_train_evaluate_api_and_resume(data, tmp_path, config):
    out = tmp_path / "run"
    for stage in ["retrieval", "ranker"]:
        train(data, out, config, stage, "cpu")
        _model, checkpoint = load_model(out / f"{stage}-best.pt")
        assert checkpoint["epoch"] == 0
        train(data, out, config, stage, "cpu", out / f"{stage}-last.pt")
    rp, mp = out / "retrieval-best.pt", out / "ranker-best.pt"
    metrics = evaluate(data, rp, mp, k=5, candidates=10)
    assert metrics["ctr"]["count"] == 50
    assert 0 <= metrics["retrieval_hit_at_k"] <= metrics["candidate_recall"] <= 1
    assert metrics["rerank_hit_at_k"] <= metrics["candidate_recall"]
    json.dumps(metrics, allow_nan=False)
    client = TestClient(create_app(data, rp, mp))
    assert client.get("/health").status_code == 200
    response = client.post(
        "/recommend",
        json={"user_id": "new-user", "history": ["0", "1"], "k": 5, "candidates": 10, "exclude_seen": True},
    )
    assert response.status_code == 200
    result = response.json()
    assert result["cold_user"] and len(result["items"]) == 5
    assert not {"0", "1"} & {r["item_id"] for r in result["items"]}
    assert all(r["p_conversion"] <= r["p_click"] for r in result["items"])
    assert client.post("/recommend", json={"user_id": "1", "k": 10, "candidates": 1}).status_code == 422
    assert client.post("/recommend", json={"user_id": "1", "history": ["x"] * 1001}).status_code == 422


def test_ann_matches_exact_and_version_guard(tmp_path, config):
    pytest.importorskip("faiss")
    model = SequenceEncoder(22, **config).eval()
    path = tmp_path / "catalog.index"
    build_ann(model, path, "test-hash")
    index = ANNIndex(path, "test-hash")
    h = torch.zeros(2, 5, dtype=torch.long)
    _, exact = exact_topk(model, h, 5)
    _, ann = index.search(model(h).detach().numpy(), 5)
    assert np.array_equal(ann, exact.numpy())
    with pytest.raises(ValueError):
        ANNIndex(path, "wrong-hash")
