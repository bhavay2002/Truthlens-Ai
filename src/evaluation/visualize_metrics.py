"""
File: src/evaluation/visualize_metrics.py

Purpose
------
Visualization utilities for model evaluation metrics.
Generates plots for:
- Confusion matrix heatmap
- ROC curve
- Precision–Recall curve
- Metric comparison bar chart
Used in TruthLens AI evaluation pipeline to support
analysis, reporting, and experiment tracking.

Inputs
-----
confusion_matrix : array-like
y_true : array-like
y_proba : array-like
evaluation_results : dict

Outputs
------
Saved visualization figures in reports/figures
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Iterable

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import auc, precision_recall_curve, roc_curve

logger = logging.getLogger(__name__)

sns.set_style("whitegrid")


def _ensure_output_dir(output_dir: str | Path) -> Path:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def _to_1d_array(values: Iterable[float], *, name: str) -> np.ndarray:
    arr = np.asarray(list(values), dtype=float)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be a 1D sequence.")
    if arr.size == 0:
        raise ValueError(f"{name} cannot be empty.")
    return arr


def plot_confusion_matrix(
    cm,
    labels=("Real", "Fake"),
    output_dir="reports/figures",
) -> Path:
    """Plot and save confusion matrix heatmap."""
    output_dir = _ensure_output_dir(output_dir)
    cm_array = np.asarray(cm)
    if cm_array.ndim != 2:
        raise ValueError("confusion matrix must be a 2D array.")

    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm_array,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels,
        ax=ax,
    )
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    ax.set_title("Confusion Matrix")
    fig.tight_layout()

    save_path = output_dir / "confusion_matrix.png"
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)

    logger.info("Saved confusion matrix plot: %s", save_path)
    return save_path


def plot_roc_curve(
    y_true,
    y_proba,
    output_dir="reports/figures",
) -> Path:
    """Plot and save ROC curve."""
    output_dir = _ensure_output_dir(output_dir)
    y_true_arr = _to_1d_array(y_true, name="y_true")
    y_proba_arr = _to_1d_array(y_proba, name="y_proba")

    if y_true_arr.shape[0] != y_proba_arr.shape[0]:
        raise ValueError("y_true and y_proba must have the same length.")
    if np.unique(y_true_arr).size < 2:
        raise ValueError("ROC curve requires at least two classes in y_true.")

    fpr, tpr, _ = roc_curve(y_true_arr, y_proba_arr)
    roc_auc = auc(fpr, tpr)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(
        fpr,
        tpr,
        label=f"ROC Curve (AUC = {roc_auc:.3f})",
    )
    ax.plot([0, 1], [0, 1], linestyle="--")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve")
    ax.legend()
    fig.tight_layout()

    save_path = output_dir / "roc_curve.png"
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)

    logger.info("Saved ROC curve: %s", save_path)
    return save_path


def plot_precision_recall_curve(
    y_true,
    y_proba,
    output_dir="reports/figures",
) -> Path:
    """Plot and save precision-recall curve."""
    output_dir = _ensure_output_dir(output_dir)
    y_true_arr = _to_1d_array(y_true, name="y_true")
    y_proba_arr = _to_1d_array(y_proba, name="y_proba")

    if y_true_arr.shape[0] != y_proba_arr.shape[0]:
        raise ValueError("y_true and y_proba must have the same length.")

    precision, recall, _ = precision_recall_curve(y_true_arr, y_proba_arr)
    pr_auc = auc(recall, precision)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(
        recall,
        precision,
        label=f"PR Curve (AUC = {pr_auc:.3f})",
    )
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curve")
    ax.legend()
    fig.tight_layout()

    save_path = output_dir / "precision_recall_curve.png"
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)

    logger.info("Saved precision-recall curve: %s", save_path)
    return save_path


def plot_metric_comparison(
    results: Dict[str, Any],
    output_dir="reports/figures",
) -> Path:
    """Plot and save summary bar chart for key metrics."""
    output_dir = _ensure_output_dir(output_dir)

    metrics = [
        "accuracy",
        "precision",
        "recall",
        "f1",
        "balanced_accuracy",
        "mcc",
    ]
    values = [float(results.get(metric, 0.0)) for metric in metrics]

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.barplot(
        x=metrics,
        y=values,
        ax=ax,
    )

    min_value = min(values) if values else 0.0
    max_value = max(values) if values else 1.0
    lower = min(0.0, min_value - 0.05)
    upper = max(1.0, max_value + 0.05)

    ax.set_ylim(lower, upper)
    ax.set_title("Model Performance Metrics")
    ax.set_ylabel("Score")
    ax.tick_params(axis="x", rotation=30)
    fig.tight_layout()

    save_path = output_dir / "metric_comparison.png"
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)

    logger.info("Saved metric comparison chart: %s", save_path)
    return save_path


def visualize_evaluation(
    results: Dict[str, Any],
    y_true=None,
    y_proba=None,
    output_dir="reports/figures",
) -> dict[str, Path]:
    """Generate and save all relevant evaluation visualizations."""
    generated_files: dict[str, Path] = {}
    logger.info("Generating evaluation visualizations")

    if "confusion_matrix" in results:
        generated_files["confusion_matrix"] = plot_confusion_matrix(
            results["confusion_matrix"],
            output_dir=output_dir,
        )

    if y_true is not None and y_proba is not None:
        generated_files["roc_curve"] = plot_roc_curve(
            y_true, y_proba, output_dir
        )
        generated_files["precision_recall_curve"] = (
            plot_precision_recall_curve(
                y_true,
                y_proba,
                output_dir,
            )
        )

    generated_files["metric_comparison"] = plot_metric_comparison(
        results, output_dir
    )

    logger.info("Evaluation visualizations complete")
    return generated_files
