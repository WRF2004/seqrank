"""Constant-memory histogram AUC (ties get half credit) and calibration."""

import numpy as np


class BinaryMetrics:
    def __init__(self, bins=4096):
        self.positive = np.zeros(bins, dtype=np.int64)
        self.negative = np.zeros(bins, dtype=np.int64)
        self.probability_sum = np.zeros(bins)
        self.loss = 0.0
        self.count = 0

    def update(self, y, p):
        y, p = np.asarray(y), np.asarray(p, dtype=np.float64)
        mask = y >= 0
        y, p = y[mask], np.clip(p[mask], 1e-7, 1 - 1e-7)
        ids = np.minimum((p * len(self.positive)).astype(int), len(self.positive) - 1)
        np.add.at(self.positive, ids, y.astype(int))
        np.add.at(self.negative, ids, 1 - y.astype(int))
        np.add.at(self.probability_sum, ids, p)
        self.loss += float((-y * np.log(p) - (1 - y) * np.log1p(-p)).sum())
        self.count += len(y)

    def result(self):
        npos, nneg = self.positive.sum(), self.negative.sum()
        auc = (
            float((self.positive * (np.cumsum(self.negative) - self.negative / 2)).sum() / (npos * nneg))
            if npos and nneg
            else None
        )
        return {
            "auc_hist4096": auc,
            "logloss": self.loss / self.count if self.count else None,
            "calibration_l1_hist4096": float(np.abs(self.probability_sum - self.positive).sum() / self.count)
            if self.count
            else None,
            "count": self.count,
            "positives": int(npos),
        }
