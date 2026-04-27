"""
File: metrics_engine.py
Location: src/evaluation/

Single source of truth for evaluation metrics.

Public API:
- ``compute_classification_metrics(y_true, y_pred, y_proba=None, ...)``
- ``compute_multilabel_metrics(y_true, y_pred, y_proba=None, ...)``
- ``compute_metrics_from_preds(y_true, y_pred, task_type, *, y_proba=None, ...)``
- ``MetricsEngine`` — multi-task orchestrator
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, Iterable, Optional

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    hamming_loss,
    log_loss,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)

logger = logging.getLogger(__name__)

EPS = 1e-12


# =========================================================
# COMMON HELPERS
# =========================================================

def _as_1d_int_array(values, *, name: str) -> np.ndarray:
    arr = np.asarray(values)

    if arr.size == 0:
        raise ValueError(f"{name} cannot be empty")

    if arr.ndim != 1:
        raise ValueError(f"{name} must be 1D (got shape {arr.shape})")

    return arr


def _as_2d_int_array(values, *, name: str) -> np.ndarray:
    arr = np.asarray(values)

    if arr.size == 0:
        raise ValueError(f"{name} cannot be empty")

    if arr.ndim != 2:
        raise ValueError(f"{name} must be 2D (got shape {arr.shape})")

    return arr


def _check_shape_match(a: np.ndarray, b: np.ndarray) -> None:
    if a.shape != b.shape:
        raise ValueError(
            f"Shape mismatch between y_true {a.shape} and y_pred {b.shape}"
        )


def _binary_proba_for_auc(y_proba: np.ndarray) -> Optional[np.ndarray]:
    """Return per-sample probability of the positive class for binary tasks."""
    if y_proba.ndim == 1:
        return y_proba
    if y_proba.ndim == 2 and y_proba.shape[1] == 2:
        return y_proba[:, 1]
    return None


# =========================================================
# CLASSIFICATION METRICS  (binary + multiclass)
# =========================================================

def compute_classification_metrics(
    y_true: Iterable,
    y_pred: Iterable,
    *,
    y_proba: Optional[Iterable] = None,
    average: Optional[str] = None,
    threshold: float = 0.5,
    confidence: Optional[Iterable] = None,
    labels: Optional[Iterable[int]] = None,
) -> Dict[str, Any]:
    """Compute standard classification metrics.

    Returns at minimum: ``accuracy, precision, recall, f1, f1_macro, f1_micro,
    f1_weighted, mcc, balanced_accuracy, confusion_matrix``. When ``y_proba`` is
    provided, ``roc_auc`` and ``log_loss`` are added (when computable).
    """
    # ``threshold`` and ``confidence`` are accepted to keep the engine API
    # uniform with the multilabel path; classification preds are already hard.
    del threshold, confidence

    y_true_arr = _as_1d_int_array(y_true, name="y_true")
    y_pred_arr = _as_1d_int_array(y_pred, name="y_pred")
    _check_shape_match(y_true_arr, y_pred_arr)

    unique_classes = np.unique(np.concatenate([y_true_arr, y_pred_arr]))
    is_binary = unique_classes.size <= 2 and set(unique_classes.tolist()).issubset({0, 1})

    chosen_average = average or ("binary" if is_binary else "macro")

    cm_labels = list(labels) if labels is not None else sorted(unique_classes.tolist())

    metrics: Dict[str, Any] = {
        "accuracy": float(accuracy_score(y_true_arr, y_pred_arr)),
        "balanced_accuracy": float(
            balanced_accuracy_score(y_true_arr, y_pred_arr)
        ),
        "precision": float(
            precision_score(
                y_true_arr,
                y_pred_arr,
                average=chosen_average,
                zero_division=0,
            )
        ),
        "recall": float(
            recall_score(
                y_true_arr,
                y_pred_arr,
                average=chosen_average,
                zero_division=0,
            )
        ),
        "f1": float(
            f1_score(
                y_true_arr,
                y_pred_arr,
                average=chosen_average,
                zero_division=0,
            )
        ),
        "f1_macro": float(
            f1_score(y_true_arr, y_pred_arr, average="macro", zero_division=0)
        ),
        "f1_micro": float(
            f1_score(y_true_arr, y_pred_arr, average="micro", zero_division=0)
        ),
        "f1_weighted": float(
            f1_score(y_true_arr, y_pred_arr, average="weighted", zero_division=0)
        ),
        "metric_average": chosen_average,
        "confusion_matrix": confusion_matrix(
            y_true_arr, y_pred_arr, labels=cm_labels
        ).tolist(),
    }

    try:
        metrics["mcc"] = float(matthews_corrcoef(y_true_arr, y_pred_arr))
    except ValueError:
        metrics["mcc"] = 0.0

    # Per-class f1 (helpful for downstream plots)
    metrics["per_class_f1"] = f1_score(
        y_true_arr,
        y_pred_arr,
        average=None,
        labels=cm_labels,
        zero_division=0,
    ).tolist()

    if y_proba is not None:
        proba_arr = np.asarray(y_proba, dtype=float)
        if proba_arr.shape[0] != y_true_arr.shape[0]:
            raise ValueError(
                f"y_proba length {proba_arr.shape[0]} does not match y_true "
                f"length {y_true_arr.shape[0]}"
            )

        try:
            if is_binary:
                positive_proba = _binary_proba_for_auc(proba_arr)
                if positive_proba is not None and len(np.unique(y_true_arr)) > 1:
                    metrics["roc_auc"] = float(
                        roc_auc_score(y_true_arr, positive_proba)
                    )
            else:
                if proba_arr.ndim == 2 and len(np.unique(y_true_arr)) > 1:
                    metrics["roc_auc"] = float(
                        roc_auc_score(
                            y_true_arr,
                            proba_arr,
                            multi_class="ovr",
                            average="macro",
                        )
                    )
        except ValueError as exc:
            logger.debug("roc_auc skipped: %s", exc)

        try:
            if proba_arr.ndim == 2:
                metrics["log_loss"] = float(
                    log_loss(y_true_arr, proba_arr, labels=cm_labels)
                )
            elif is_binary:
                metrics["log_loss"] = float(log_loss(y_true_arr, proba_arr))
        except ValueError as exc:
            logger.debug("log_loss skipped: %s", exc)

    return metrics


# =========================================================
# MULTILABEL METRICS
# =========================================================

def compute_multilabel_metrics(
    y_true: Iterable,
    y_pred: Iterable,
    *,
    y_proba: Optional[Iterable] = None,
    threshold: float = 0.5,
) -> Dict[str, Any]:
    """Compute multilabel classification metrics.

    Always returns: ``subset_accuracy, element_accuracy, f1_micro, f1_macro,
    f1_samples, f1_weighted, hamming_loss, per_label_f1``.
    Adds ``log_loss`` and ``roc_auc`` when ``y_proba`` is supplied.
    """
    y_true_arr = _as_2d_int_array(y_true, name="y_true")
    y_pred_arr = _as_2d_int_array(y_pred, name="y_pred")
    _check_shape_match(y_true_arr, y_pred_arr)

    metrics: Dict[str, Any] = {
        "subset_accuracy": float(
            np.all(y_true_arr == y_pred_arr, axis=1).mean()
        ),
        "element_accuracy": float((y_true_arr == y_pred_arr).mean()),
        "hamming_loss": float(hamming_loss(y_true_arr, y_pred_arr)),
        "f1_micro": float(
            f1_score(y_true_arr, y_pred_arr, average="micro", zero_division=0)
        ),
        "f1_macro": float(
            f1_score(y_true_arr, y_pred_arr, average="macro", zero_division=0)
        ),
        "f1_samples": float(
            f1_score(y_true_arr, y_pred_arr, average="samples", zero_division=0)
        ),
        "f1_weighted": float(
            f1_score(y_true_arr, y_pred_arr, average="weighted", zero_division=0)
        ),
        "per_label_f1": f1_score(
            y_true_arr, y_pred_arr, average=None, zero_division=0
        ).tolist(),
        "threshold": float(threshold),
    }

    if y_proba is not None:
        proba_arr = np.asarray(y_proba, dtype=float)
        if proba_arr.shape != y_true_arr.shape:
            raise ValueError(
                f"y_proba shape {proba_arr.shape} does not match y_true "
                f"shape {y_true_arr.shape}"
            )

        try:
            proba_clipped = np.clip(proba_arr, EPS, 1.0 - EPS)
            metrics["log_loss"] = float(
                -np.mean(
                    y_true_arr * np.log(proba_clipped)
                    + (1 - y_true_arr) * np.log(1 - proba_clipped)
                )
            )
        except ValueError as exc:
            logger.debug("multilabel log_loss skipped: %s", exc)

        try:
            valid_labels = np.where(y_true_arr.sum(axis=0) > 0)[0]
            if valid_labels.size:
                metrics["roc_auc_macro"] = float(
                    roc_auc_score(
                        y_true_arr[:, valid_labels],
                        proba_arr[:, valid_labels],
                        average="macro",
                    )
                )
        except ValueError as exc:
            logger.debug("multilabel roc_auc skipped: %s", exc)

    return metrics


# =========================================================
# UNIFIED ENTRY (used by Evaluator / EvaluationEngine)
# =========================================================

def compute_metrics_from_preds(
    y_true: Iterable,
    y_pred: Iterable,
    *,
    task_type: str,
    y_proba: Optional[Iterable] = None,
    threshold: float = 0.5,
    average: Optional[str] = None,
) -> Dict[str, Any]:
    """Route to the correct metric computer based on ``task_type``."""
    if task_type in ("binary", "multiclass", "classification"):
        return compute_classification_metrics(
            y_true,
            y_pred,
            y_proba=y_proba,
            average=average,
            threshold=threshold,
        )

    if task_type == "multilabel":
        return compute_multilabel_metrics(
            y_true,
            y_pred,
            y_proba=y_proba,
            threshold=threshold,
        )

    raise ValueError(f"Unknown task_type: {task_type!r}")


# =========================================================
# CONFIG + MULTI-TASK ENGINE
# =========================================================

@dataclass
class MetricsEngineConfig:
    default_threshold: float = 0.5
    enable_confidence_weighting: bool = False
    return_per_task: bool = True
    aggregate: bool = True


class MetricsEngine:
    """Stateless multi-task metric orchestrator."""

    # The set of metrics that we average across tasks. Adding extras is safe
    # but we keep the list explicit so downstream consumers know what to expect.
    _AGG_KEYS = (
        "accuracy",
        "balanced_accuracy",
        "f1",
        "f1_macro",
        "f1_micro",
        "f1_weighted",
        "f1_samples",
        "subset_accuracy",
        "element_accuracy",
        "hamming_loss",
        "mcc",
        "roc_auc",
        "roc_auc_macro",
        "log_loss",
    )

    def __init__(self, config: Optional[MetricsEngineConfig] = None):
        self.config = config or MetricsEngineConfig()
        logger.info("MetricsEngine initialized")

    # -----------------------------------------------------
    # SINGLE TASK
    # -----------------------------------------------------

    def compute_task(
        self,
        *,
        y_true,
        y_pred,
        y_proba=None,
        task_type: str,
        threshold: Optional[float] = None,
        confidence=None,
    ) -> Dict[str, Any]:
        del confidence  # accepted for API compatibility
        threshold = threshold if threshold is not None else self.config.default_threshold

        return compute_metrics_from_preds(
            y_true=y_true,
            y_pred=y_pred,
            y_proba=y_proba,
            task_type=task_type,
            threshold=threshold,
        )

    # -----------------------------------------------------
    # MULTI TASK
    # -----------------------------------------------------

    def compute_multitask(
        self,
        *,
        predictions: Dict[str, Dict[str, Any]],
        task_types: Dict[str, str],
        thresholds: Optional[Dict[str, float]] = None,
    ) -> Dict[str, Any]:
        results: Dict[str, Any] = {}

        for task, data in predictions.items():
            if task not in task_types:
                logger.warning("Missing task type for %s; skipping", task)
                continue

            if "y_true" not in data or "y_pred" not in data:
                logger.warning("Missing y_true/y_pred for %s; skipping", task)
                continue

            threshold = (
                thresholds.get(task)
                if thresholds and task in thresholds
                else None
            )

            try:
                results[task] = self.compute_task(
                    y_true=data["y_true"],
                    y_pred=data["y_pred"],
                    y_proba=data.get("y_proba"),
                    task_type=task_types[task],
                    threshold=threshold,
                )
            except ValueError as exc:
                logger.warning("Metrics failed for %s: %s", task, exc)

        if self.config.aggregate and results:
            results["__aggregate__"] = self._aggregate(results)

        return results

    # -----------------------------------------------------
    # AGGREGATION (mean across tasks per metric)
    # -----------------------------------------------------

    def _aggregate(self, per_task: Dict[str, Dict[str, Any]]) -> Dict[str, float]:
        agg: Dict[str, float] = {}

        for key in self._AGG_KEYS:
            values = [
                float(metrics[key])
                for metrics in per_task.values()
                if isinstance(metrics, dict)
                and isinstance(metrics.get(key), (int, float))
            ]

            if values:
                agg[key] = float(np.mean(values))

        agg["num_tasks"] = float(len(per_task))
        return agg


__all__ = [
    "MetricsEngine",
    "MetricsEngineConfig",
    "compute_classification_metrics",
    "compute_metrics_from_preds",
    "compute_multilabel_metrics",
]
