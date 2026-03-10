"""
Model Evaluation Module for TruthLens AI
Provides comprehensive evaluation metrics and reporting
"""

import logging
import json
from pathlib import Path
from typing import Dict, Any, Optional

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
    matthews_corrcoef
)

from src.utils.config_loader import get_path, load_config

logger = logging.getLogger(__name__)


# -------------------------------------------------
# Evaluation Function
# -------------------------------------------------

def evaluate(
    y_true,
    y_pred,
    y_proba: Optional[np.ndarray] = None
) -> Dict[str, Any]:
    """
    Comprehensive model evaluation.

    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        y_proba: Predicted probabilities for positive class

    Returns:
        dict containing evaluation metrics
    """

    try:

        y_true = np.array(y_true)
        y_pred = np.array(y_pred)

        results = {}

        # -------------------------------------------------
        # Core Metrics
        # -------------------------------------------------

        results["accuracy"] = accuracy_score(y_true, y_pred)
        results["balanced_accuracy"] = balanced_accuracy_score(y_true, y_pred)

        results["precision"] = precision_score(
            y_true,
            y_pred,
            average="binary",
            zero_division=0
        )

        results["recall"] = recall_score(
            y_true,
            y_pred,
            average="binary",
            zero_division=0
        )

        results["f1"] = f1_score(
            y_true,
            y_pred,
            average="binary",
            zero_division=0
        )

        results["mcc"] = matthews_corrcoef(y_true, y_pred)

        # -------------------------------------------------
        # Confusion Matrix
        # -------------------------------------------------

        cm = confusion_matrix(y_true, y_pred)

        results["confusion_matrix"] = cm.tolist()

        # -------------------------------------------------
        # ROC AUC
        # -------------------------------------------------

        if y_proba is not None:

            y_proba = np.array(y_proba)

            results["roc_auc"] = roc_auc_score(y_true, y_proba)

            fpr, tpr, thresholds = roc_curve(y_true, y_proba)

            results["roc_curve"] = {
                "fpr": fpr.tolist(),
                "tpr": tpr.tolist(),
                "thresholds": thresholds.tolist()
            }

        # -------------------------------------------------
        # Classification Report
        # -------------------------------------------------

        report_dict = classification_report(
            y_true,
            y_pred,
            target_names=["Real", "Fake"],
            output_dict=True
        )

        results["classification_report"] = report_dict

        # -------------------------------------------------
        # Dataset Statistics
        # -------------------------------------------------

        results["dataset_stats"] = {
            "total_samples": int(len(y_true)),
            "real_samples": int(np.sum(y_true == 0)),
            "fake_samples": int(np.sum(y_true == 1))
        }

        # -------------------------------------------------
        # Logging
        # -------------------------------------------------

        logger.info("================================================")
        logger.info("Model Evaluation Results")
        logger.info("================================================")

        logger.info(f"Accuracy: {results['accuracy']:.4f}")
        logger.info(f"Balanced Accuracy: {results['balanced_accuracy']:.4f}")
        logger.info(f"Precision: {results['precision']:.4f}")
        logger.info(f"Recall: {results['recall']:.4f}")
        logger.info(f"F1 Score: {results['f1']:.4f}")
        logger.info(f"MCC: {results['mcc']:.4f}")

        if "roc_auc" in results:
            logger.info(f"ROC-AUC: {results['roc_auc']:.4f}")

        logger.info(f"Dataset size: {results['dataset_stats']['total_samples']}")

        logger.info("================================================")

        return results

    except Exception as e:
        logger.error(f"Error during evaluation: {e}")
        raise


# -------------------------------------------------
# Save Results
# -------------------------------------------------

def save_evaluation_results(
    results: Dict[str, Any],
    output_path: str | Path | None = None
):
    """
    Save evaluation results to JSON file.
    """

    try:

        if output_path is None:
            config = load_config()
            output_path = get_path(
                config,
                "paths",
                "evaluation_results_path",
                default="reports/evaluation_results.json",
            )
        else:
            output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Convert numpy objects safely
        def convert(obj):

            if isinstance(obj, np.ndarray):
                return obj.tolist()

            if isinstance(obj, (np.float32, np.float64)):
                return float(obj)

            if isinstance(obj, (np.int32, np.int64)):
                return int(obj)

            return obj

        serializable_results = json.loads(
            json.dumps(results, default=convert)
        )

        with output_path.open("w", encoding="utf-8") as f:
            json.dump(serializable_results, f, indent=4)

        logger.info(f"Evaluation results saved to: {output_path}")

    except Exception as e:
        logger.error(f"Error saving evaluation results: {e}")
        raise
