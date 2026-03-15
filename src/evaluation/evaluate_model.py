"""
File: src/evaluation/evaluate_model.py
Purpose
------
Provides evaluation utilities for classification models.
Features

--------- 
Accuracy, Precision, Recall, F1
- Balanced accuracy
- Matthews Correlation Coefficient
- Confusion matrix
- ROC-AUC and ROC curve
- Classification report
- Dataset statistics
- JSON report export

Used for evaluating fake news classifiers in TruthLens AI.

Inputs
-----
y_true : array-like
y_pred : array-like
y_proba : Optional[array-like]

Outputs
------
Dictionary containing evaluation metrics
Saved JSON evaluation report
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_auc_score,
    roc_curve,
    balanced_accuracy_score,
    matthews_corrcoef,
)

from src.utils.settings import load_settings

logger = logging.getLogger(__name__)


def _prepare_probability_vector(
    y_proba: np.ndarray | list[float] | list[list[float]],
    n_samples: int,
) -> np.ndarray:
    """
    Convert probabilities to a 1D positive-class probability vector.

    Accepts:
    - shape: (n_samples,)
    - shape: (n_samples, 2) where column 1 is the positive class
    """
    proba = np.asarray(y_proba)

    if proba.ndim == 2:
        if proba.shape[1] < 2:
            raise ValueError(
                "y_proba with 2 dimensions must contain at least two columns."
            )
        proba = proba[:, 1]
    elif proba.ndim != 1:
        raise ValueError(
            "y_proba must be a 1D vector or a 2D probability matrix."
        )

    if proba.shape[0] != n_samples:
        raise ValueError(
            "y_proba length must match y_true/y_pred length "
            f"({proba.shape[0]} != {n_samples})."
        )

    return proba.astype(float)


def evaluate(
    y_true,
    y_pred,
    y_proba: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """
    Evaluate classification model predictions.

    Parameters
    ----------
    y_true : array-like
        Ground truth labels.

    y_pred : array-like
        Model predicted labels.

    y_proba : Optional[np.ndarray]
        Predicted probabilities for positive class.

    Returns
    -------
    Dict[str, Any]
        Dictionary containing evaluation metrics.
    """

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    if y_true.size == 0:
        raise ValueError("y_true cannot be empty.")
    if y_true.shape[0] != y_pred.shape[0]:
        raise ValueError(
            "y_true and y_pred must have the same length "
            f"({y_true.shape[0]} != {y_pred.shape[0]})."
        )

    results: Dict[str, Any] = {}

    results["accuracy"] = accuracy_score(y_true, y_pred)
    results["balanced_accuracy"] = balanced_accuracy_score(y_true, y_pred)
    results["precision"] = precision_score(
        y_true, y_pred, average="binary", zero_division=0
    )
    results["recall"] = recall_score(
        y_true, y_pred, average="binary", zero_division=0
    )
    results["f1"] = f1_score(y_true, y_pred, average="binary", zero_division=0)
    results["mcc"] = matthews_corrcoef(y_true, y_pred)

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    results["confusion_matrix"] = cm.tolist()

    if y_proba is not None:
        proba_vector = _prepare_probability_vector(
            y_proba, n_samples=len(y_true)
        )

        if np.unique(y_true).size > 1:
            results["roc_auc"] = roc_auc_score(y_true, proba_vector)
            fpr, tpr, thresholds = roc_curve(y_true, proba_vector)
            results["roc_curve"] = {
                "fpr": fpr.tolist(),
                "tpr": tpr.tolist(),
                "thresholds": thresholds.tolist(),
            }
        else:
            logger.warning(
                "Skipping ROC metrics because y_true contains only one class."
            )

    results["classification_report"] = classification_report(
        y_true,
        y_pred,
        labels=[0, 1],
        target_names=["Real", "Fake"],
        output_dict=True,
        zero_division=0,
    )

    results["dataset_stats"] = {
        "total_samples": int(len(y_true)),
        "real_samples": int(np.sum(y_true == 0)),
        "fake_samples": int(np.sum(y_true == 1)),
    }

    logger.info("================================================")
    logger.info("Model Evaluation Results")
    logger.info("================================================")
    logger.info("Accuracy: %.4f", results["accuracy"])
    logger.info("Balanced Accuracy: %.4f", results["balanced_accuracy"])
    logger.info("Precision: %.4f", results["precision"])
    logger.info("Recall: %.4f", results["recall"])
    logger.info("F1 Score: %.4f", results["f1"])
    logger.info("MCC: %.4f", results["mcc"])
    if "roc_auc" in results:
        logger.info("ROC-AUC: %.4f", results["roc_auc"])
    logger.info("Dataset size: %s", results["dataset_stats"]["total_samples"])
    logger.info("================================================")

    return results


# -------------------------------------------------
# Save Evaluation Results
# -------------------------------------------------


def save_evaluation_results(
    results: Dict[str, Any],
    output_path: str | Path | None = None,
) -> None:
    """
    Save evaluation results to JSON.

    Parameters
    ----------
    results : Dict[str, Any]
        Evaluation results dictionary.

    output_path : Optional[str | Path]
        Path to save results.
    """

    if output_path is None:
        output_path = load_settings().paths.evaluation_results_path
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    def convert(obj: Any) -> Any:
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.generic):
            return obj.item()
        return obj

    serializable_results = json.loads(json.dumps(results, default=convert))

    with output_path.open("w", encoding="utf-8") as output_file:
        json.dump(serializable_results, output_file, indent=4)

    logger.info("Evaluation results saved to: %s", output_path)
