from __future__ import annotations

import logging
from typing import Dict, Any, List, Optional

import numpy as np


logger = logging.getLogger(__name__)


EPS = 1e-12


# =========================================================
# BASIC STATISTICS
# =========================================================

def compute_basic_stats(values: np.ndarray) -> Dict[str, float]:
    if values.size == 0:
        return {}

    return {
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
        "median": float(np.median(values)),
    }


# =========================================================
# HISTOGRAM
# =========================================================

def compute_histogram(values: np.ndarray, bins: int = 10) -> Dict[str, Any]:
    hist, bin_edges = np.histogram(values, bins=bins, range=(0.0, 1.0))

    return {
        "counts": hist.tolist(),
        "bin_edges": bin_edges.tolist(),
    }


# =========================================================
# CALIBRATION METRICS
# =========================================================

def expected_calibration_error(
    probs: np.ndarray,
    labels: np.ndarray,
    n_bins: int = 10,
) -> float:
    """
    ECE: measures calibration quality
    """

    if probs.ndim == 2:
        confidences = np.max(probs, axis=1)
        predictions = np.argmax(probs, axis=1)
    else:
        confidences = probs
        predictions = (probs >= 0.5).astype(int)

    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0

    for i in range(n_bins):
        mask = (confidences >= bins[i]) & (confidences < bins[i + 1])
        if not np.any(mask):
            continue

        acc = np.mean(predictions[mask] == labels[mask])
        conf = np.mean(confidences[mask])

        ece += np.abs(acc - conf) * np.sum(mask) / len(probs)

    return float(ece)


# =========================================================
# BRIER SCORE
# =========================================================

def brier_score(probs: np.ndarray, labels: np.ndarray) -> float:
    """
    Measures probabilistic accuracy
    """

    if probs.ndim == 2:
        one_hot = np.eye(probs.shape[1])[labels]
        return float(np.mean((probs - one_hot) ** 2))

    return float(np.mean((probs - labels) ** 2))


# =========================================================
# DRIFT DETECTION (KL DIVERGENCE)
# =========================================================

def kl_divergence(p: np.ndarray, q: np.ndarray) -> float:
    p = np.clip(p, EPS, 1.0)
    q = np.clip(q, EPS, 1.0)
    return float(np.sum(p * np.log(p / q)))


def compute_distribution_shift(
    reference: np.ndarray,
    current: np.ndarray,
    bins: int = 20,
) -> float:
    """
    Compare distributions using KL divergence
    """

    ref_hist, _ = np.histogram(reference, bins=bins, range=(0, 1), density=True)
    cur_hist, _ = np.histogram(current, bins=bins, range=(0, 1), density=True)

    ref_hist /= np.sum(ref_hist) + EPS
    cur_hist /= np.sum(cur_hist) + EPS

    return kl_divergence(ref_hist, cur_hist)


# =========================================================
# TASK-LEVEL METRICS
# =========================================================

def compute_task_metrics(
    scores: Dict[str, float],
) -> Dict[str, Any]:
    """
    Metrics for single sample
    """

    values = np.array(list(scores.values()), dtype=np.float32)

    return {
        "stats": compute_basic_stats(values),
    }


def compute_batch_metrics(
    batch_scores: List[Dict[str, float]],
) -> Dict[str, Any]:
    """
    Metrics across batch
    """

    if not batch_scores:
        return {}

    keys = batch_scores[0].keys()

    aggregated: Dict[str, List[float]] = {k: [] for k in keys}

    for sample in batch_scores:
        for k in keys:
            aggregated[k].append(sample[k])

    results = {}

    for k, vals in aggregated.items():
        arr = np.array(vals, dtype=np.float32)

        results[k] = {
            "stats": compute_basic_stats(arr),
            "histogram": compute_histogram(arr),
        }

    return results


# =========================================================
# SYSTEM METRICS
# =========================================================

class AggregationMetrics:
    """
    Central metrics collector for aggregation pipeline
    """

    def __init__(self) -> None:
        self.history: List[Dict[str, Any]] = []

    def update(self, scores: Dict[str, float]) -> None:
        self.history.append(scores)

    def summarize(self) -> Dict[str, Any]:
        return compute_batch_metrics(self.history)

    def reset(self) -> None:
        self.history.clear()

    def size(self) -> int:
        return len(self.history)