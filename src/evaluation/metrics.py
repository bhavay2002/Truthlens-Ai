"""
File Name: metrics.py
Module: TruthLens AI - Metrics
Description:
    Centralized metric computation module used across the TruthLens AI
    evaluation system. Supports binary, multi-class, and multi-label
    classification metrics with robust validation and structured outputs.
Dependencies:
    numpy
    sklearn.metrics
    logging
Inputs:
    y_true: Ground truth labels
    y_pred: Predicted labels
    y_proba: Optional predicted probabilities
Outputs:
    Dictionary containing computed evaluation metrics
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
)

logger = logging.getLogger(__name__)


def _validate_shapes(
    y_true: Iterable,
    y_pred: Iterable
) -> None:
    """
    Validate shapes of true and predicted labels.
    """

    y_true_arr = np.asarray(y_true)
    y_pred_arr = np.asarray(y_pred)

    if y_true_arr.shape[0] == 0:
        raise ValueError("y_true cannot be empty.")

    if y_true_arr.shape != y_pred_arr.shape:
        raise ValueError(
            f"Shape mismatch: y_true {y_true_arr.shape} vs y_pred {y_pred_arr.shape}"
        )


def compute_classification_metrics(
    y_true: Iterable,
    y_pred: Iterable,
    y_proba: Optional[Iterable] = None
) -> Dict[str, Any]:
    """
    Compute metrics for binary or multi-class classification tasks.
    """

    logger.info("Computing classification metrics")

    _validate_shapes(y_true, y_pred)

    y_true_arr = np.asarray(y_true)
    y_pred_arr = np.asarray(y_pred)

    n_classes = np.unique(y_true_arr).size
    average = "binary" if n_classes == 2 else "macro"
    results: Dict[str, Any] = {}

    try:
        results["accuracy"] = float(accuracy_score(y_true_arr, y_pred_arr))
        results["balanced_accuracy"] = float(
            balanced_accuracy_score(y_true_arr, y_pred_arr)
        )

        results["precision"] = float(
            precision_score(
                y_true_arr,
                y_pred_arr,
                average=average,
                zero_division=0
            )
        )

        results["recall"] = float(
            recall_score(
                y_true_arr,
                y_pred_arr,
                average=average,
                zero_division=0
            )
        )

        results["f1"] = float(
            f1_score(
                y_true_arr,
                y_pred_arr,
                average=average,
                zero_division=0
            )
        )

        results["mcc"] = float(matthews_corrcoef(y_true_arr, y_pred_arr))

        results["confusion_matrix"] = confusion_matrix(
            y_true_arr,
            y_pred_arr
        ).tolist()

        if y_proba is not None:
            try:
                y_proba_arr = np.asarray(y_proba)

                if y_proba_arr.ndim == 1:
                    if n_classes != 2:
                        logger.warning(
                            "Skipping roc_auc: 1D probabilities provided for non-binary labels."
                        )
                    else:
                        results["roc_auc"] = float(
                            roc_auc_score(y_true_arr, y_proba_arr)
                        )
                else:
                    results["roc_auc_ovr"] = float(
                        roc_auc_score(
                            y_true_arr,
                            y_proba_arr,
                            multi_class="ovr"
                        )
                    )
            except Exception as exc:
                logger.warning("ROC AUC computation failed: %s", str(exc))

    except Exception as exc:
        logger.exception("Classification metric computation failed")
        raise RuntimeError("Failed to compute classification metrics") from exc

    return results


def compute_multilabel_metrics(
    y_true: Iterable,
    y_pred: Iterable
) -> Dict[str, Any]:
    """
    Compute metrics for multi-label classification tasks.
    """

    logger.info("Computing multilabel metrics")

    y_true_arr = np.asarray(y_true)
    y_pred_arr = np.asarray(y_pred)

    if y_true_arr.shape != y_pred_arr.shape:
        raise ValueError(
            f"Shape mismatch: y_true {y_true_arr.shape} vs y_pred {y_pred_arr.shape}"
        )

    if y_true_arr.shape[0] == 0:
        raise ValueError("y_true cannot be empty.")

    results: Dict[str, Any] = {}

    try:
        results["subset_accuracy"] = float(
            np.mean(np.all(y_true_arr == y_pred_arr, axis=1))
        )

        results["element_accuracy"] = float(
            np.mean(y_true_arr == y_pred_arr)
        )

        results["f1_micro"] = float(
            f1_score(y_true_arr, y_pred_arr, average="micro", zero_division=0)
        )

        results["f1_macro"] = float(
            f1_score(y_true_arr, y_pred_arr, average="macro", zero_division=0)
        )

        results["f1_weighted"] = float(
            f1_score(y_true_arr, y_pred_arr, average="weighted", zero_division=0)
        )

        results["precision_micro"] = float(
            precision_score(y_true_arr, y_pred_arr, average="micro", zero_division=0)
        )

        results["recall_micro"] = float(
            recall_score(y_true_arr, y_pred_arr, average="micro", zero_division=0)
        )

        results["hamming_loss"] = float(
            hamming_loss(y_true_arr, y_pred_arr)
        )

        results["jaccard_micro"] = float(
            jaccard_score(y_true_arr, y_pred_arr, average="micro", zero_division=0)
        )

    except Exception as exc:
        logger.exception("Multilabel metric computation failed")
        raise RuntimeError("Failed to compute multilabel metrics") from exc

    return results