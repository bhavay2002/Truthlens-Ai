"""
File Name: evaluate_model.py
Module: TruthLens AI - Lightweight Evaluation API
Description:
    Backward-compatible evaluation entrypoint used by tests and simple
    consumers. Computes classification metrics and includes dataset summary
    statistics in a stable output schema.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, Optional

import numpy as np

from .metrics import compute_classification_metrics


def _to_serializable_label(label: Any) -> Any:
    """Convert numpy scalar labels into plain Python serializable values."""

    if isinstance(label, np.generic):
        return label.item()
    return label


def evaluate(
    y_true: Iterable,
    y_pred: Iterable,
    y_proba: Optional[Iterable] = None,
) -> Dict[str, Any]:
    """
    Evaluate classification predictions and return metrics plus dataset stats.
    """

    y_true_arr = np.asarray(y_true)
    y_pred_arr = np.asarray(y_pred)

    if y_true_arr.shape[0] == 0:
        raise ValueError("y_true cannot be empty.")

    if y_true_arr.shape != y_pred_arr.shape:
        raise ValueError(
            f"Shape mismatch: y_true {y_true_arr.shape} vs y_pred {y_pred_arr.shape}"
        )

    metrics = compute_classification_metrics(
        y_true=y_true_arr,
        y_pred=y_pred_arr,
        y_proba=y_proba,
    )

    labels, counts = np.unique(y_true_arr, return_counts=True)
    num_classes = int(labels.size)

    metrics["metric_average"] = "macro" if num_classes > 2 else "binary"
    metrics["dataset_stats"] = {
        "num_samples": int(y_true_arr.shape[0]),
        "num_classes": num_classes,
        "class_counts": {
            str(_to_serializable_label(label)): int(count)
            for label, count in zip(labels, counts)
        },
    }

    return metrics
