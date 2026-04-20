from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable, Tuple

import numpy as np
import matplotlib.pyplot as plt
import torch

from src.models.calibration import CalibrationMetricConfig, CalibrationMetrics

logger = logging.getLogger(__name__)


# =========================================================
# VALIDATION
# =========================================================

def _validate_inputs(
    y_true: Iterable,
    probs: Iterable
) -> Tuple[np.ndarray, np.ndarray]:

    labels = np.asarray(y_true)

    if labels.ndim != 1:
        raise ValueError("y_true must be 1D")

    if labels.size == 0:
        raise ValueError("y_true cannot be empty")

    if not np.issubdtype(labels.dtype, np.integer):
        raise ValueError("y_true must contain integer class labels")

    labels = labels.astype(np.int64)

    probs_arr = np.asarray(probs, dtype=np.float64)

    if probs_arr.shape[0] != labels.shape[0]:
        raise ValueError("y_true and probs must have same length")

    # sanitize probabilities
    probs_arr = np.nan_to_num(probs_arr, nan=0.0, posinf=1.0, neginf=0.0)

    if np.any(probs_arr < 0) or np.any(probs_arr > 1):
        raise ValueError("Probabilities must be within [0, 1]")

    return labels, probs_arr


def _validate_bins(n_bins: int) -> None:
    if not isinstance(n_bins, int) or n_bins <= 0:
        raise ValueError("n_bins must be a positive integer")


def _auto_bins(n_samples: int) -> int:
    """
    Adaptive bin selection for ECE.

    - Lower bound: 5 bins
    - Upper bound: 30 bins
    - Uses sqrt heuristic for stability
    """
    return max(5, min(int(np.sqrt(n_samples)), 30))


# =========================================================
# PROBABILITY NORMALIZATION
# =========================================================

def _normalize_probs(probs: np.ndarray) -> np.ndarray:
    if probs.ndim == 2:
        row_sums = probs.sum(axis=1, keepdims=True)
        row_sums = np.where(row_sums == 0, 1.0, row_sums)
        probs = probs / row_sums
    return probs


def _to_probs_2d(labels: np.ndarray, probs_arr: np.ndarray) -> np.ndarray:

    if probs_arr.ndim == 1:
        if np.unique(labels).size > 2:
            raise ValueError("1D probabilities only valid for binary classification")
        probs_arr = np.stack([1.0 - probs_arr, probs_arr], axis=1)

    elif probs_arr.ndim != 2:
        raise ValueError("probs must be 1D or 2D")

    probs_arr = _normalize_probs(probs_arr)

    n_classes = np.unique(labels).size

    if probs_arr.shape[1] < n_classes:
        raise ValueError("Probability columns fewer than number of label classes")

    return probs_arr


# =========================================================
# ECE
# =========================================================

def expected_calibration_error(
    y_true: Iterable,
    probs: Iterable,
    n_bins: int | None = None
) -> float:

    labels, probs_arr = _validate_inputs(y_true, probs)

    if n_bins is None:
        n_bins = _auto_bins(len(labels))
    else:
        _validate_bins(n_bins)

    probs_2d = _to_probs_2d(labels, probs_arr)

    metric = CalibrationMetrics(CalibrationMetricConfig(n_bins=n_bins))

    ece = metric.expected_calibration_error(
        torch.from_numpy(probs_2d.astype(np.float32)),
        torch.from_numpy(labels),
    )

    # ensure float output
    if isinstance(ece, torch.Tensor):
        ece = float(ece.item())

    logger.info("Expected Calibration Error (ECE): %.6f", ece)

    return float(ece)


# =========================================================
# RELIABILITY DIAGRAM
# =========================================================

def plot_reliability_diagram(
    y_true: Iterable,
    probs: Iterable,
    save_path: str | Path,
    n_bins: int | None = None
) -> Path:

    labels, probs_arr = _validate_inputs(y_true, probs)

    if n_bins is None:
        n_bins = _auto_bins(len(labels))
    else:
        _validate_bins(n_bins)

    probs_2d = _to_probs_2d(labels, probs_arr)

    metric = CalibrationMetrics(CalibrationMetricConfig(n_bins=n_bins))

    stats = metric.reliability_statistics(
        torch.from_numpy(probs_2d.astype(np.float32)),
        torch.from_numpy(labels),
    )

    prob_true = np.asarray(stats["bin_accuracy"], dtype=np.float64)
    prob_pred = np.asarray(stats["bin_confidence"], dtype=np.float64)

    # remove NaNs (empty bins)
    mask = np.isfinite(prob_true) & np.isfinite(prob_pred)
    prob_true = prob_true[mask]
    prob_pred = prob_pred[mask]

    output_path = Path(save_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # isolated figure (no global side effects)
    fig, ax = plt.subplots(figsize=(6, 6))

    ax.plot(prob_pred, prob_true, marker="o", label="Model")
    ax.plot([0, 1], [0, 1], linestyle="--", label="Perfect Calibration")

    ax.set_xlabel("Confidence")
    ax.set_ylabel("Accuracy")
    ax.set_title("Reliability Diagram")
    ax.legend()

    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)

    logger.info("Reliability diagram saved to %s", output_path)

    return output_path