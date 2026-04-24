"""
File: metrics.py (FINAL - RESEARCH + INDUSTRY GRADE)
"""

from __future__ import annotations

import logging
from typing import Dict, Any, Iterable, Optional

import numpy as np

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    balanced_accuracy_score,
    matthews_corrcoef,
    confusion_matrix,
    roc_auc_score,
    hamming_loss,
    jaccard_score,
    log_loss,
)

logger = logging.getLogger(__name__)


# =========================================================
# VALIDATION
# =========================================================
def _validate_shapes(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    if y_true.shape[0] == 0:
        raise ValueError("Empty input")

    if y_true.shape != y_pred.shape:
        raise ValueError("Shape mismatch")


# =========================================================
# CONFUSION BREAKDOWN (NEW 🔥)
# =========================================================
def _confusion_breakdown(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)

    breakdown = {}
    n_classes = cm.shape[0]

    for i in range(n_classes):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp
        tn = cm.sum() - (tp + fp + fn)

        breakdown[f"class_{i}"] = {
            "tp": int(tp),
            "fp": int(fp),
            "fn": int(fn),
            "tn": int(tn),
        }

    return {
        "matrix": cm.tolist(),
        "per_class": breakdown,
    }


# =========================================================
# CLASSIFICATION METRICS
# =========================================================
def compute_classification_metrics(
    y_true: Iterable,
    y_pred: Iterable,
    y_proba: Optional[Iterable] = None,
) -> Dict[str, Any]:

    _validate_shapes(y_true, y_pred)

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    n_classes = np.unique(y_true).size
    avg = "binary" if n_classes == 2 else "macro"

    results: Dict[str, Any] = {}

    # ---------------------------
    # CORE
    # ---------------------------
    results["accuracy"] = float(accuracy_score(y_true, y_pred))
    results["balanced_accuracy"] = float(balanced_accuracy_score(y_true, y_pred))

    results["precision"] = float(precision_score(y_true, y_pred, average=avg, zero_division=0))
    results["recall"] = float(recall_score(y_true, y_pred, average=avg, zero_division=0))
    results["f1"] = float(f1_score(y_true, y_pred, average=avg, zero_division=0))

    # 🔥 NEW (imbalance safe)
    results["precision_weighted"] = float(precision_score(y_true, y_pred, average="weighted", zero_division=0))
    results["recall_weighted"] = float(recall_score(y_true, y_pred, average="weighted", zero_division=0))
    results["f1_weighted"] = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))

    results["mcc"] = float(matthews_corrcoef(y_true, y_pred))

    # 🔥 NEW structured confusion
    results["confusion"] = _confusion_breakdown(y_true, y_pred)

    # ---------------------------
    # PER-CLASS
    # ---------------------------
    results["per_class_precision"] = precision_score(
        y_true, y_pred, average=None, zero_division=0
    ).tolist()

    results["per_class_recall"] = recall_score(
        y_true, y_pred, average=None, zero_division=0
    ).tolist()

    results["per_class_f1"] = f1_score(
        y_true, y_pred, average=None, zero_division=0
    ).tolist()

    # ---------------------------
    # PROBABILITY METRICS
    # ---------------------------
    if y_proba is not None:
        try:
            y_proba = np.asarray(y_proba)

            if y_proba.ndim == 1 and n_classes == 2:
                results["roc_auc"] = float(roc_auc_score(y_true, y_proba))
                results["log_loss"] = float(log_loss(y_true, y_proba))

            elif y_proba.ndim == 2:
                results["roc_auc_ovr"] = float(
                    roc_auc_score(y_true, y_proba, multi_class="ovr")
                )
                results["log_loss"] = float(log_loss(y_true, y_proba))

        except Exception as e:
            logger.warning(f"Probability metrics failed: {e}")

    return results


# =========================================================
# THRESHOLD TUNING (NEW 🔥)
# =========================================================
def _optimize_threshold(y_true, y_proba):
    best_t = 0.5
    best_f1 = -1

    for t in np.linspace(0.1, 0.9, 17):
        pred = (y_proba > t).astype(int)
        score = f1_score(y_true, pred, average="micro")

        if score > best_f1:
            best_f1 = score
            best_t = t

    return best_t


# =========================================================
# MULTILABEL METRICS (UPGRADED)
# =========================================================
def compute_multilabel_metrics(
    y_true: Iterable,
    y_pred: Iterable,
    y_proba: Optional[Iterable] = None,
    threshold: float = 0.5,
    auto_threshold: bool = False,
) -> Dict[str, Any]:

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    if y_true.shape != y_pred.shape:
        raise ValueError("Shape mismatch")

    results: Dict[str, Any] = {}

    # ---------------------------
    # THRESHOLD TUNING
    # ---------------------------
    if auto_threshold and y_proba is not None:
        threshold = _optimize_threshold(y_true, y_proba)
        results["optimized_threshold"] = float(threshold)

        y_pred = (y_proba > threshold).astype(int)

    # ---------------------------
    # BASIC
    # ---------------------------
    results["subset_accuracy"] = float(np.mean(np.all(y_true == y_pred, axis=1)))
    results["element_accuracy"] = float(np.mean(y_true == y_pred))

    # ---------------------------
    # METRICS
    # ---------------------------
    results["f1_micro"] = float(f1_score(y_true, y_pred, average="micro", zero_division=0))
    results["f1_macro"] = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    results["f1_weighted"] = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))

    results["precision_micro"] = float(precision_score(y_true, y_pred, average="micro", zero_division=0))
    results["recall_micro"] = float(recall_score(y_true, y_pred, average="micro", zero_division=0))

    results["hamming_loss"] = float(hamming_loss(y_true, y_pred))
    results["jaccard_micro"] = float(jaccard_score(y_true, y_pred, average="micro", zero_division=0))

    # ---------------------------
    # PROBABILITY METRICS
    # ---------------------------
    if y_proba is not None:
        y_proba = np.asarray(y_proba)

        try:
            results["roc_auc_macro"] = float(
                roc_auc_score(y_true, y_proba, average="macro")
            )

        except Exception as e:
            logger.warning(f"Multilabel AUROC failed: {e}")

    return results