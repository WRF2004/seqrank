"""One feature transform for training rows, candidate ranking and API requests."""
from __future__ import annotations

import numpy as np
import torch

DENSE_NAMES = ['log_duration', 'duration_missing', 'log_history_length', 'mean_log_watch',
               'mean_log_duration', 'log_last_gap_seconds', 'hour_sin', 'hour_cos']


def dense_features(duration: np.ndarray, timestamp: np.ndarray, history: np.ndarray,
                   history_watch: np.ndarray, history_duration: np.ndarray,
                   history_time: np.ndarray) -> np.ndarray:
    """Inputs duration/time [B], histories [B,L]; output float32 [B,8]. No current labels."""
    mask = history != 0
    count = mask.sum(-1)
    def mean_log(values):
        valid = mask & (values >= 0)
        return (np.log1p(np.maximum(values, 0)) * valid).sum(-1) / valid.sum(-1).clip(1)
    last = np.where(mask, history_time, 0).max(-1)
    gap = np.where(count > 0, np.maximum(timestamp - last, 0) / 1000, 0)
    angle = ((timestamp // 1000) % 86400) / 86400 * (2 * np.pi)
    return np.stack([np.log1p(np.maximum(duration, 0)), duration <= 0, np.log1p(count),
                     mean_log(history_watch), mean_log(history_duration), np.log1p(gap),
                     np.sin(angle), np.cos(angle)], -1).astype(np.float32)


def transform_batch(batch: dict, statistics: dict) -> dict:
    """Return copied dict with standardized dense input; masks retain raw semantics."""
    result = dict(batch)
    mean = batch['dense'].new_tensor(statistics['dense_mean'])
    scale = batch['dense'].new_tensor(statistics['dense_std'])
    result['dense'] = ((batch['dense'].float() - mean) / scale).clamp(-10, 10)
    return result


def candidate_batch(batch: dict, items: torch.Tensor, catalog: dict, statistics: dict) -> dict:
    """items [B,C] -> flattened [B*C,...]; recompute candidate-dependent features only."""
    b, c = items.shape
    result = {}
    for key in ('user', 'history', 'history_watch', 'history_duration', 'history_time', 'timestamp'):
        v = batch[key]
        result[key] = v[:, None].expand((b, c) + v.shape[1:]).reshape((b*c,) + v.shape[1:])
    result['item'] = items.reshape(-1)
    ids = result['item'].detach().cpu().numpy()
    result['duration'] = torch.as_tensor(np.asarray(catalog['duration'][ids]).copy(), device=items.device)
    result['category'] = torch.as_tensor(np.asarray(catalog['category'][ids]).copy(), device=items.device).long()
    # dense is unstandardized in dataset/API; use the same function as preparation.
    array = dense_features(result['duration'].cpu().numpy(), result['timestamp'].cpu().numpy(),
                           result['history'].cpu().numpy(), result['history_watch'].cpu().numpy(),
                           result['history_duration'].cpu().numpy(), result['history_time'].cpu().numpy())
    result['dense'] = torch.from_numpy(array).to(items.device)
    return transform_batch(result, statistics)
