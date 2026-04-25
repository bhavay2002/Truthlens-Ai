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

EPS = 1e-12


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
# CONFUSION
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
# BRIER SCORE
# =========================================================

def brier_score(y_true, y_proba):
    y_true = np.asarray(y_true)
    y_proba = np.asarray(y_proba)

    if y_proba.ndim == 1:
        return float(np.mean((y_proba - y_true) ** 2))

    one_hot = np.eye(y_proba.shape[1])[y_true]
    return float(np.mean((y_proba - one_hot) ** 2))


# =========================================================
#  APPLY THRESHOLD (NEW)
# =========================================================

def _apply_threshold(y_proba, threshold):
    y_proba = np.asarray(y_proba)

    if y_proba.ndim == 1:
        return (y_proba >= threshold).astype(int)

    # binary in shape (N,2)
    if y_proba.shape[1] == 2:
        return (y_proba[:, 1] >= threshold).astype(int)

    # multilabel
    return (y_proba >= threshold).astype(int)


# =========================================================
#  CONFIDENCE WEIGHTING (NEW)
# =========================================================

def _confidence_weighted_metric(metric_fn, y_true, y_pred, confidence):
    try:
        weights = np.asarray(confidence)
        return float(metric_fn(y_true, y_pred, sample_weight=weights))
    except Exception:
        return float(metric_fn(y_true, y_pred))


# =========================================================
# CLASSIFICATION METRICS (UPGRADED)
# =========================================================

def compute_classification_metrics(
    y_true: Iterable,
    y_pred: Iterable,
    y_proba: Optional[Iterable] = None,
    threshold: Optional[float] = None,
    confidence: Optional[Iterable] = None,
) -> Dict[str, Any]:

    _validate_shapes(y_true, y_pred)

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    #  threshold override
    if threshold is not None and y_proba is not None:
        try:
            y_pred = _apply_threshold(y_proba, threshold)
        except Exception as e:
            logger.warning(f"Thresholding failed: {e}")

    results: Dict[str, Any] = {}

    # ---------------------------
    # CORE
    # ---------------------------
    results["accuracy"] = float(accuracy_score(y_true, y_pred))
    results["balanced_accuracy"] = float(balanced_accuracy_score(y_true, y_pred))

    results["precision_macro"] = float(
        precision_score(y_true, y_pred, average="macro", zero_division=0)
    )
    results["recall_macro"] = float(
        recall_score(y_true, y_pred, average="macro", zero_division=0)
    )
    results["f1_macro"] = float(
        f1_score(y_true, y_pred, average="macro", zero_division=0)
    )

    results["precision_weighted"] = float(
        precision_score(y_true, y_pred, average="weighted", zero_division=0)
    )
    results["recall_weighted"] = float(
        recall_score(y_true, y_pred, average="weighted", zero_division=0)
    )
    results["f1_weighted"] = float(
        f1_score(y_true, y_pred, average="weighted", zero_division=0)
    )

    results["mcc"] = float(matthews_corrcoef(y_true, y_pred))

    # ---------------------------
    # CONFIDENCE-WEIGHTED METRICS
    # ---------------------------
    if confidence is not None:
        results["accuracy_weighted"] = _confidence_weighted_metric(
            accuracy_score, y_true, y_pred, confidence
        )
        results["f1_weighted_conf"] = _confidence_weighted_metric(
            lambda yt, yp, sample_weight=None: f1_score(
                yt, yp, average="weighted", zero_division=0, sample_weight=sample_weight
            ),
            y_true,
            y_pred,
            confidence,
        )

    # ---------------------------
    # CONFUSION
    # ---------------------------
    results["confusion"] = _confusion_breakdown(y_true, y_pred)

    # ---------------------------
    # PER CLASS
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
        y_proba = np.asarray(y_proba)

        try:
            if y_proba.ndim == 1 or y_proba.shape[1] == 1:
                y_proba = y_proba.reshape(-1)

                results["roc_auc"] = float(
                    roc_auc_score(y_true, y_proba)
                )
                results["log_loss"] = float(log_loss(y_true, y_proba))
                results["brier"] = brier_score(y_true, y_proba)

            else:
                results["roc_auc_ovr"] = float(
                    roc_auc_score(y_true, y_proba, multi_class="ovr")
                )
                results["log_loss"] = float(log_loss(y_true, y_proba))
                results["brier"] = brier_score(y_true, y_proba)

        except Exception as e:
            logger.warning(f"Probability metrics failed: {e}")

    return results


# =========================================================
# MULTILABEL METRICS (UPGRADED)
# =========================================================

def compute_multilabel_metrics(
    y_true: Iterable,
    y_pred: Iterable,
    y_proba: Optional[Iterable] = None,
    threshold: float = 0.5,
) -> Dict[str, Any]:

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    if y_true.shape != y_pred.shape:
        raise ValueError("Shape mismatch")

    results: Dict[str, Any] = {}

    results["subset_accuracy"] = float(np.mean(np.all(y_true == y_pred, axis=1)))
    results["element_accuracy"] = float(np.mean(y_true == y_pred))

    results["f1_micro"] = float(f1_score(y_true, y_pred, average="micro", zero_division=0))
    results["f1_macro"] = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    results["f1_weighted"] = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))

    results["precision_micro"] = float(precision_score(y_true, y_pred, average="micro", zero_division=0))
    results["recall_micro"] = float(recall_score(y_true, y_pred, average="micro", zero_division=0))

    results["hamming_loss"] = float(hamming_loss(y_true, y_pred))
    results["jaccard_micro"] = float(jaccard_score(y_true, y_pred, average="micro", zero_division=0))

    # ---------------------------
    # PER LABEL
    # ---------------------------
    results["per_label_f1"] = f1_score(
        y_true, y_pred, average=None, zero_division=0
    ).tolist()

    results["per_label_precision"] = precision_score(
        y_true, y_pred, average=None, zero_division=0
    ).tolist()

    results["per_label_recall"] = recall_score(
        y_true, y_pred, average=None, zero_division=0
    ).tolist()

    # ---------------------------
    # PROBABILITY METRICS
    # ---------------------------
    if y_proba is not None:
        y_proba = np.asarray(y_proba)

        try:
            results["roc_auc_macro"] = float(
                roc_auc_score(y_true, y_proba, average="macro")
            )

            results["brier"] = float(np.mean((y_proba - y_true) ** 2))

        except Exception as e:
            logger.warning(f"Multilabel AUROC failed: {e}")

    return results