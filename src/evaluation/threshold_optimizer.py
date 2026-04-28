from __future__ import annotations

import logging
from typing import Any, Dict, Iterable, Literal, Optional

import numpy as np
from sklearn.metrics import (
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
)

from src.config.task_config import get_task_type

logger = logging.getLogger(__name__)

EPS = 1e-12


# =========================================================
# BINARY THRESHOLD (vectorized)
# =========================================================

def optimize_binary_threshold(
    y_true: Iterable,
    probs: Iterable,
    *,
    metric: Literal["f1", "precision", "recall"] = "f1",
) -> Dict[str, float]:
    y_true = np.asarray(y_true).astype(int).reshape(-1)
    probs = np.asarray(probs, dtype=float).reshape(-1)

    if y_true.shape != probs.shape:
        raise ValueError("y_true and probs must have the same length")

    if len(np.unique(y_true)) < 2:
        return {"threshold": 0.5, "score": 0.0, "metric": metric}

    precision, recall, thresholds = precision_recall_curve(y_true, probs)

    # Drop the trailing point that has no associated threshold.
    precision = precision[:-1]
    recall = recall[:-1]

    if metric == "f1":
        scores = 2 * precision * recall / (precision + recall + EPS)
    elif metric == "precision":
        scores = precision
    elif metric == "recall":
        scores = recall
    else:
        raise ValueError(f"Unsupported metric: {metric}")

    if scores.size == 0:
        return {"threshold": 0.5, "score": 0.0, "metric": metric}

    best_idx = int(np.argmax(scores))
    best_t = float(thresholds[best_idx])
    best_score = float(scores[best_idx])

    return {"threshold": best_t, "score": best_score, "metric": metric}


# =========================================================
# MULTILABEL THRESHOLD
# =========================================================

def _score_per_label(
    y_label: np.ndarray, p_label: np.ndarray, metric: str
) -> Dict[str, float]:
    if len(np.unique(y_label)) < 2:
        return {"threshold": 0.5, "score": 0.0}

    precision, recall, thresholds = precision_recall_curve(y_label, p_label)
    precision = precision[:-1]
    recall = recall[:-1]

    if metric == "f1":
        scores = 2 * precision * recall / (precision + recall + EPS)
    elif metric == "precision":
        scores = precision
    elif metric == "recall":
        scores = recall
    else:
        raise ValueError(f"Unsupported metric: {metric}")

    if scores.size == 0:
        return {"threshold": 0.5, "score": 0.0}

    best_idx = int(np.argmax(scores))
    return {"threshold": float(thresholds[best_idx]), "score": float(scores[best_idx])}


def optimize_multilabel_thresholds(
    y_true: Iterable,
    probs: Iterable,
    *,
    metric: Literal["f1", "precision", "recall"] = "f1",
    strategy: Literal["per_label", "global"] = "per_label",
) -> Dict[str, Any]:
    y_true = np.asarray(y_true).astype(int)
    probs = np.asarray(probs, dtype=float)

    if y_true.shape != probs.shape:
        raise ValueError("y_true and probs must have the same shape")

    n_labels = y_true.shape[1]

    if strategy == "global":
        # HIGH E9: vectorize the global multilabel sweep. The previous
        # implementation called ``f1_score`` once per candidate threshold
        # (T iterations × O(N·L) each) which dominated eval time on large
        # multilabel sets. Build the (N, L, T) prediction tensor once and
        # reduce TP/FP/FN along the sample axis.
        candidates = np.linspace(0.05, 0.95, 19)
        # Shapes: probs (N, L), candidates (T,) -> preds (N, L, T)
        preds = (probs[:, :, None] >= candidates[None, None, :]).astype(np.int8)
        y_true_3d = y_true[:, :, None].astype(np.int8)

        tp = (preds * y_true_3d).sum(axis=0)              # (L, T)
        fp = (preds * (1 - y_true_3d)).sum(axis=0)         # (L, T)
        fn = ((1 - preds) * y_true_3d).sum(axis=0)         # (L, T)

        if metric == "f1":
            scores_per_label = (2 * tp) / (2 * tp + fp + fn + EPS)   # (L, T)
        elif metric == "precision":
            scores_per_label = tp / (tp + fp + EPS)                  # (L, T)
        else:  # recall
            scores_per_label = tp / (tp + fn + EPS)                  # (L, T)

        macro = scores_per_label.mean(axis=0)              # (T,)
        best_idx = int(np.argmax(macro))
        return {
            "strategy": "global",
            "threshold": float(candidates[best_idx]),
            "score": float(macro[best_idx]),
        }

    thresholds_out: list[float] = []
    scores_out: list[float] = []

    for i in range(n_labels):
        result = _score_per_label(y_true[:, i], probs[:, i], metric)
        thresholds_out.append(result["threshold"])
        scores_out.append(result["score"])

    return {
        "strategy": "per_label",
        "thresholds": thresholds_out,
        "scores": scores_out,
        "mean_score": float(np.mean(scores_out)) if scores_out else 0.0,
    }


# =========================================================
# UNIFIED API
# =========================================================

def optimize_thresholds(
    y_true: Iterable,
    probs: Iterable,
    *,
    task: Optional[str] = None,
    metric: str = "f1",
    strategy: str = "per_label",
) -> Dict[str, Any]:
    probs_arr = np.asarray(probs, dtype=float)
    task_type = get_task_type(task) if task else None

    if task_type == "multiclass":
        # Threshold optimization is not meaningful for argmax decisions.
        return {
            "task_type": "multiclass",
            "skipped": True,
            "reason": "argmax decision rule",
        }

    if task_type == "binary" or (task_type is None and probs_arr.ndim == 1):
        if probs_arr.ndim == 2 and probs_arr.shape[1] == 2:
            probs_arr = probs_arr[:, 1]
        return optimize_binary_threshold(y_true, probs_arr, metric=metric)

    if task_type == "multilabel" or (task_type is None and probs_arr.ndim == 2):
        return optimize_multilabel_thresholds(
            y_true, probs_arr, metric=metric, strategy=strategy
        )

    raise ValueError(f"Unsupported task_type: {task_type}")


# =========================================================
# CLASS WRAPPER
# =========================================================

class ThresholdOptimizer:
    """Stateless wrapper around the threshold optimization helpers."""

    def __init__(self, *, metric: str = "f1", strategy: str = "per_label"):
        self.metric = metric
        self.strategy = strategy

    def optimize(self, collected: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        if not isinstance(collected, dict):
            return None

        y_true = collected.get("y_true")
        y_proba = collected.get("y_proba")

        if y_true is None or y_proba is None:
            return None

        task_type = collected.get("task_type")
        task = collected.get("task")

        return optimize_thresholds(
            y_true=y_true,
            probs=y_proba,
            task=task if task_type is None else None,
            metric=self.metric,
            strategy=self.strategy,
        )


__all__ = [
    "ThresholdOptimizer",
    "optimize_binary_threshold",
    "optimize_multilabel_thresholds",
    "optimize_thresholds",
]
