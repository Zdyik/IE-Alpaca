"""Metrics for one prediction per validation vehicle."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    recall_score,
    roc_auc_score,
)


def score_binary(labels, probabilities, threshold: float = 0.5) -> dict:
    y = np.asarray(labels, dtype=int)
    p = np.asarray(probabilities, dtype=float)
    if y.ndim != 1 or p.ndim != 1 or len(y) != len(p) or not len(y):
        raise ValueError("标签与预测概率必须是等长非空一维数组")
    if not set(np.unique(y)).issubset({0, 1}) or not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
        raise ValueError("标签须为 0/1，概率须在 [0,1] 且有限")
    pred = (p >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {
        "n": int(len(y)), "positive": int(y.sum()), "negative": int(len(y) - y.sum()),
        "roc_auc": float(roc_auc_score(y, p)) if len(np.unique(y)) == 2 else None,
        "pr_auc": float(average_precision_score(y, p)) if y.sum() else None,
        "brier": float(brier_score_loss(y, p)),
        "accuracy_at_0_5": float(accuracy_score(y, p >= 0.5)),
        "accuracy_at_selected_threshold": float(accuracy_score(y, pred)),
        "selected_threshold": float(threshold),
        "threshold_rule": "fixed_0.5",
        "recall": float(recall_score(y, pred, zero_division=0)),
        "specificity": float(tn / (tn + fp)) if tn + fp else None,
        "f1": float(f1_score(y, pred, zero_division=0)),
        "confusion": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }


def auc_interval(labels, probabilities, seed: int, samples: int = 1000) -> dict:
    y = np.asarray(labels, dtype=int)
    p = np.asarray(probabilities, dtype=float)
    if len(np.unique(y)) != 2:
        return {"lower": None, "upper": None, "valid_bootstraps": 0}
    rng = np.random.default_rng(seed)
    scores = []
    for _ in range(samples):
        indices = rng.integers(0, len(y), size=len(y))
        if len(np.unique(y[indices])) == 2:
            scores.append(roc_auc_score(y[indices], p[indices]))
    if not scores:
        return {"lower": None, "upper": None, "valid_bootstraps": 0}
    low, high = np.quantile(scores, [0.025, 0.975])
    return {"lower": float(low), "upper": float(high), "valid_bootstraps": len(scores)}
