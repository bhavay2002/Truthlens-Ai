from __future__ import annotations

import logging
from typing import Dict, Any, Optional, Literal

import numpy as np
import matplotlib.pyplot as plt

from src.config.task_config import get_task_type

logger = logging.getLogger(__name__)

EPS = 1e-12


# =========================================================
# BINNING
# =========================================================

def _bin_stats(confidence, correctness, n_bins: int):
    """
    Compute bin-wise accuracy and confidence.
    """
    bins = np.linspace(0.0, 1.0, n_bins + 1)

    bin_ids = np.digitize(confidence, bins) - 1
    bin_ids = np.clip(bin_ids, 0, n_bins - 1)

    acc = np.zeros(n_bins)
    conf = np.zeros(n_bins)
    counts = np.zeros(n_bins)

    for b in range(n_bins):
        idx = bin_ids == b
        if np.any(idx):
            counts[b] = np.sum(idx)
            acc[b] = np.mean(correctness[idx])
            conf[b] = np.mean(confidence[idx])

    return {
        "accuracy": acc,
        "confidence": conf,
        "counts": counts,
        "bin_centers": (bins[:-1] + bins[1:]) / 2,
    }


# =========================================================
# BINARY RELIABILITY
# =========================================================

def _binary_reliability(y_true, probs, n_bins):
    probs = probs.reshape(-1)
    preds = (probs >= 0.5).astype(int)
    correctness = (preds == y_true).astype(float)

    return _bin_stats(probs, correctness, n_bins)


# =========================================================
# MULTICLASS RELIABILITY
# =========================================================

def _multiclass_reliability(y_true, probs, n_bins):

    preds = np.argmax(probs, axis=1)
    confidence = np.max(probs, axis=1)
    correctness = (preds == y_true).astype(float)

    return _bin_stats(confidence, correctness, n_bins)


# =========================================================
# PER-CLASS RELIABILITY
# =========================================================

def _per_class_reliability(y_true, probs, n_bins):

    n_classes = probs.shape[1]
    results = {}

    for c in range(n_classes):
        y_bin = (y_true == c).astype(int)
        p_c = probs[:, c]

        results[f"class_{c}"] = _bin_stats(p_c, y_bin, n_bins)

    return results


# =========================================================
# MULTILABEL RELIABILITY
# =========================================================

def _multilabel_reliability(y_true, probs, n_bins):

    n_labels = probs.shape[1]
    results = {}

    for i in range(n_labels):
        y_i = y_true[:, i]
        p_i = probs[:, i]

        results[f"label_{i}"] = _bin_stats(p_i, y_i, n_bins)

    return results


# =========================================================
# PLOTTING
# =========================================================

def _plot_curve(bin_data, title="Reliability Diagram", save_path=None):

    fig, ax = plt.subplots()

    ax.plot([0, 1], [0, 1], linestyle="--")  # perfect calibration

    ax.plot(
        bin_data["confidence"],
        bin_data["accuracy"],
        marker="o",
    )

    ax.set_xlabel("Confidence")
    ax.set_ylabel("Accuracy")
    ax.set_title(title)

    if save_path:
        fig.savefig(save_path)

    return fig


# =========================================================
# MAIN API
# =========================================================

def reliability_diagram(
    y_true,
    probs,
    *,
    task: Optional[str] = None,
    n_bins: int = 10,
    mode: Literal["global", "per_class"] = "global",
    save_path: Optional[str] = None,
) -> Dict[str, Any]:

    y_true = np.asarray(y_true)
    probs = np.asarray(probs)

    task_type = get_task_type(task) if task else None

    logger.info(f"[RELIABILITY] task={task} type={task_type}")

    # -----------------------------------------------------
    # GLOBAL
    # -----------------------------------------------------
    if task_type == "binary":
        stats = _binary_reliability(y_true, probs, n_bins)
        fig = _plot_curve(stats, "Binary Reliability", save_path)

        return {"global": stats, "figure": fig}

    elif task_type == "multiclass":

        global_stats = _multiclass_reliability(y_true, probs, n_bins)
        fig = _plot_curve(global_stats, "Multiclass Reliability", save_path)

        result = {"global": global_stats, "figure": fig}

        if mode == "per_class":
            result["per_class"] = _per_class_reliability(y_true, probs, n_bins)

        return result

    elif task_type == "multilabel":

        per_label = _multilabel_reliability(y_true, probs, n_bins)

        return {
            "per_label": per_label,
        }

    else:
        raise ValueError(f"Unsupported task_type: {task_type}")

# =========================================================
# CLASS WRAPPER (used by src.evaluation.calibration)
# =========================================================

class ReliabilityDiagram:
    """Lightweight OO wrapper around the functional reliability_diagram API."""

    def __init__(self, n_bins: int = 10):
        self.n_bins = n_bins

    def compute(self, probs, y_true, task_type: Optional[str] = None,
                save_path: Optional[str] = None, mode: str = "global"):
        try:
            return reliability_diagram(
                y_true=y_true,
                probs=probs,
                task_type=task_type,
                n_bins=self.n_bins,
                save_path=save_path,
                mode=mode,
            )
        except TypeError:
            # Fallback for unknown task type — compute multiclass stats directly.
            try:
                stats = _multiclass_reliability(y_true, probs, self.n_bins)
                return {"global": stats}
            except Exception as e:
                logger.warning("ReliabilityDiagram.compute fallback failed: %s", e)
                return {}
