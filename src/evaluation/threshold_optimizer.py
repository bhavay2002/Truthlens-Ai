from __future__ import annotations

import logging
from typing import Dict, Iterable, Optional, Literal

import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score

from src.config.task_config import get_task_type

logger = logging.getLogger(__name__)

EPS = 1e-12


# =========================================================
# METRIC SELECTION
# =========================================================

def _compute_metric(y_true, y_pred, metric: str):

    if metric == "f1":
        return f1_score(y_true, y_pred, average="binary")

    elif metric == "precision":
        return precision_score(y_true, y_pred, average="binary")

    elif metric == "recall":
        return recall_score(y_true, y_pred, average="binary")

    else:
        raise ValueError(f"Unsupported metric: {metric}")


# =========================================================
# BINARY THRESHOLD OPTIMIZATION
# =========================================================

def optimize_binary_threshold(
    y_true: Iterable,
    probs: Iterable,
    *,
    metric: Literal["f1", "precision", "recall"] = "f1",
    thresholds: Optional[np.ndarray] = None,
) -> Dict[str, float]:

    y_true = np.asarray(y_true)
    probs = np.asarray(probs).reshape(-1)

    if thresholds is None:
        thresholds = np.linspace(0.01, 0.99, 99)

    best_t = 0.5
    best_score = -1.0

    for t in thresholds:
        preds = (probs >= t).astype(int)
        score = _compute_metric(y_true, preds, metric)

        if score > best_score:
            best_score = score
            best_t = t

    return {
        "threshold": float(best_t),
        "score": float(best_score),
        "metric": metric,
    }


# =========================================================
# MULTILABEL THRESHOLD OPTIMIZATION
# =========================================================

def optimize_multilabel_thresholds(
    y_true: Iterable,
    probs: Iterable,
    *,
    metric: Literal["f1", "precision", "recall"] = "f1",
    thresholds: Optional[np.ndarray] = None,
    strategy: Literal["per_label", "global"] = "per_label",
) -> Dict[str, Any]:

    y_true = np.asarray(y_true)
    probs = np.asarray(probs)

    n_labels = y_true.shape[1]

    if thresholds is None:
        thresholds = np.linspace(0.01, 0.99, 99)

    # -----------------------------------------------------
    # GLOBAL THRESHOLD
    # -----------------------------------------------------
    if strategy == "global":

        best_t = 0.5
        best_score = -1.0

        for t in thresholds:
            preds = (probs >= t).astype(int)

            score = f1_score(
                y_true,
                preds,
                average="macro" if metric == "f1" else "micro",
                zero_division=0,
            )

            if score > best_score:
                best_score = score
                best_t = t

        return {
            "strategy": "global",
            "threshold": float(best_t),
            "score": float(best_score),
        }

    # -----------------------------------------------------
    # PER LABEL THRESHOLDS
    # -----------------------------------------------------
    thresholds_out = []
    scores_out = []

    for i in range(n_labels):

        y_i = y_true[:, i]
        p_i = probs[:, i]

        best_t = 0.5
        best_score = -1.0

        for t in thresholds:
            preds = (p_i >= t).astype(int)
            score = _compute_metric(y_i, preds, metric)

            if score > best_score:
                best_score = score
                best_t = t

        thresholds_out.append(best_t)
        scores_out.append(best_score)

    return {
        "strategy": "per_label",
        "thresholds": np.array(thresholds_out),
        "scores": np.array(scores_out),
        "mean_score": float(np.mean(scores_out)),
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

    probs = np.asarray(probs)

    task_type = get_task_type(task) if task else None

    # -------------------------
    # AUTO DETECT
    # -------------------------
    if task_type == "binary" or probs.ndim == 1:
        return optimize_binary_threshold(
            y_true,
            probs,
            metric=metric,
        )

    elif task_type == "multilabel" or probs.ndim == 2:
        return optimize_multilabel_thresholds(
            y_true,
            probs,
            metric=metric,
            strategy=strategy,
        )

    else:
        raise ValueError(f"Unsupported task_type: {task_type}")