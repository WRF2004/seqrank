import math

import torch
from torch import nn
from torch.nn import functional as F


class SequenceEncoder(nn.Module):
    """SASRec-inspired next-click encoder; right padded with a learned BOS.

    BOS ensures empty histories and padding queries always have an unmasked key.
    This is not a bit-for-bit reproduction of the original SASRec architecture.
    """

    def __init__(self, n_items, dim=64, max_len=50, heads=2, layers=2, dropout=0.1, **_):
        super().__init__()
        self.max_len = max_len
        self.item = nn.Embedding(n_items, dim, padding_idx=0)
        self.position = nn.Embedding(max_len + 1, dim)
        self.bos = nn.Parameter(torch.randn(1, 1, dim) * 0.02)
        layer = nn.TransformerEncoderLayer(
            dim, heads, 4 * dim, dropout, batch_first=True, norm_first=True, activation="gelu"
        )
        self.transformer = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(dim)

    def sequence(self, history):
        b, length = history.shape
        x = torch.cat([self.bos.expand(b, -1, -1), self.item(history)], dim=1)
        x = x * math.sqrt(x.size(-1)) + self.position(torch.arange(length + 1, device=x.device))
        padding = torch.cat([torch.zeros(b, 1, dtype=torch.bool, device=x.device), history == 0], 1)
        causal = torch.ones(length + 1, length + 1, dtype=torch.bool, device=x.device).triu(1)
        return self.norm(self.transformer(x, mask=causal, src_key_padding_mask=padding))

    def forward(self, history):
        h = self.sequence(history)
        return h[torch.arange(h.size(0), device=h.device), (history != 0).sum(1)]


def sampled_loss(model, batch, negatives=64):
    """Uniform sampled binary objective, not full-softmax likelihood.

    Target collisions are deterministically shifted (uniform over all other IDs).
    Earlier clicked items are NOT excluded: repeated consumption is allowed.
    Unknown items are masked from the retrieval objective.
    """
    q = model(batch["history"])
    target = batch["item"]
    raw_model = model.module if hasattr(model, "module") else model
    n = raw_model.item.num_embeddings
    raw = torch.randint(2, n - 1, (q.size(0), negatives), device=q.device)
    safe_target = target.clamp_min(2)
    neg = raw + (raw >= safe_target[:, None]).long()
    pos_score = (q * raw_model.item(safe_target)).sum(-1) / math.sqrt(q.size(-1))
    neg_score = torch.einsum("bd,bnd->bn", q, raw_model.item(neg)) / math.sqrt(q.size(-1))
    loss = F.softplus(-pos_score) + F.softplus(neg_score).mean(1)
    mask = ((batch["click"] == 1) & (target >= 2)).float()
    # Fixed denominator permits zero-positive batches under DDP without skipped collectives.
    return (loss * mask).mean()


class Ranker(nn.Module):
    """DIN-style target attention + MMoE + ESMM-style CTR*CVR joint head."""

    def __init__(self, n_items, n_users, dim=64, experts=4, dropout=0.1, **_):
        super().__init__()
        self.item = nn.Embedding(n_items, dim, padding_idx=0)
        self.user = nn.Embedding(n_users, dim, padding_idx=0)
        self.attention = nn.Sequential(nn.Linear(dim * 4, dim), nn.PReLU(), nn.Linear(dim, 1))
        width = dim * 4
        self.experts = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(width, 2 * dim),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Linear(2 * dim, dim),
                    nn.ReLU(),
                )
                for _ in range(experts)
            ]
        )
        self.gates = nn.ModuleList([nn.Linear(width, experts) for _ in range(3)])
        self.towers = nn.ModuleList(
            [nn.Sequential(nn.Linear(dim, dim), nn.ReLU(), nn.Linear(dim, 1)) for _ in range(3)]
        )

    def forward(self, user, history, item):
        target = self.item(item)
        h = self.item(history)
        t = target[:, None].expand_as(h)
        logits = self.attention(torch.cat([h, t, h - t, h * t], -1)).squeeze(-1)
        valid = history != 0
        # Finite sentinel plus renormalization makes all-padding histories exactly zero.
        weights = logits.masked_fill(~valid, -1e4).softmax(-1) * valid
        weights = weights / weights.sum(-1, keepdim=True).clamp_min(1e-8)
        interest = (weights[..., None] * h).sum(1)
        x = torch.cat([self.user(user), target, interest, target * interest], -1)
        experts = torch.stack([expert(x) for expert in self.experts], 1)
        values = [
            tower((gate(x).softmax(-1)[..., None] * experts).sum(1)).squeeze(-1)
            for gate, tower in zip(self.gates, self.towers)
        ]
        ctr, cvr = values[0].sigmoid(), values[1].sigmoid()
        return {
            "ctr_logit": values[0],
            "cvr_logit": values[1],
            "ctr": ctr,
            "cvr": cvr,
            "ctcvr": ctr * cvr,
            "log_dwell": F.softplus(values[2]),
        }


def rank_loss(output, batch, conversion_weight=1.0, dwell_weight=0.1):
    click = F.binary_cross_entropy_with_logits(output["ctr_logit"], batch["click"])
    # Stable log(sigmoid(CTR)*sigmoid(CVR)) in fp32, including under autocast.
    log_joint = F.logsigmoid(output["ctr_logit"].float()) + F.logsigmoid(output["cvr_logit"].float())
    log_joint = log_joint.clamp_max(-1e-7)
    y = batch["conversion"].clamp_min(0)
    conv_losses = -y * log_joint - (1 - y) * torch.log(-torch.expm1(log_joint))
    conv_mask = (batch["conversion"] >= 0).float()
    dwell_mask = ((batch["click"] == 1) & (batch["dwell"] >= 0)).float()
    dwell_losses = F.smooth_l1_loss(
        output["log_dwell"].float(), batch["dwell"].clamp_min(0).log1p(), reduction="none"
    )
    # Fixed denominator yields identical weighting across DDP ranks.
    return (
        click
        + conversion_weight * (conv_losses * conv_mask).mean()
        + dwell_weight * (dwell_losses * dwell_mask).mean()
    )
