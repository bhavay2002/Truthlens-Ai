"""
File Name: visualize.py
Module: Visualization - Evaluation and Analysis Plots
Description:
    Comprehensive visualization utilities for the TruthLens AI system.
    This module provides research-grade visualization functions used
    for model evaluation, training diagnostics, feature analysis, and
    embedding inspection.

    Implemented visualizations:
        - Confusion Matrix
        - ROC Curve
        - Precision–Recall Curve
        - Calibration Curve
        - Training Curves
        - Feature Importance
        - Embedding Projection (PCA / t-SNE)

Dependencies:
    logging
    typing
    pathlib
    numpy
    matplotlib
    seaborn
    sklearn
Inputs:
    Model predictions, labels, feature scores, embeddings
Outputs:
    Matplotlib figures and axes objects
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable, List, Tuple, Dict, Optional

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import (
    roc_curve,
    auc,
    precision_recall_curve,
)
from sklearn.calibration import calibration_curve
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------
# Utility Functions
# ---------------------------------------------------------------------

def _ensure_numpy(x: Iterable) -> np.ndarray:
    """Convert input to numpy array."""
    return np.asarray(x)


def _save_figure(fig: plt.Figure, save_path: Optional[str | Path]) -> None:
    """Save figure if path is provided."""
    if save_path:
        path = Path(save_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=300, bbox_inches="tight")
        logger.info("Figure saved to %s", path)


# ---------------------------------------------------------------------
# Confusion Matrix
# ---------------------------------------------------------------------

def plot_confusion_matrix(
    cm: np.ndarray,
    labels: List[str] | None = None,
    normalize: bool = False,
    cmap: str = "Blues",
    save_path: str | Path | None = None,
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Plot confusion matrix heatmap.
    """

    cm = _ensure_numpy(cm)

    if cm.ndim != 2 or cm.shape[0] != cm.shape[1]:
        raise ValueError("Confusion matrix must be square")

    if normalize:
        row_sums = cm.sum(axis=1, keepdims=True).astype(float)
        cm = np.where(row_sums == 0, 0.0, cm.astype(float) / np.where(row_sums == 0, 1.0, row_sums))

    if labels is None:
        labels = [str(i) for i in range(cm.shape[0])]

    fig, ax = plt.subplots(figsize=(6, 5))

    sns.heatmap(
        cm,
        annot=True,
        fmt=".2f" if normalize else "g",
        cmap=cmap,
        xticklabels=labels,
        yticklabels=labels,
        cbar=False,
        ax=ax,
    )

    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion Matrix")

    fig.tight_layout()

    _save_figure(fig, save_path)

    return fig, ax


# ---------------------------------------------------------------------
# ROC Curve
# ---------------------------------------------------------------------

def plot_roc_curve(
    y_true: Iterable,
    y_scores: Iterable,
    save_path: str | Path | None = None,
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Plot ROC curve and AUC score.
    """

    y_true = _ensure_numpy(y_true)
    y_scores = _ensure_numpy(y_scores)

    fpr, tpr, _ = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)

    fig, ax = plt.subplots()

    ax.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
    ax.plot([0, 1], [0, 1], linestyle="--")

    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve")
    ax.legend()

    fig.tight_layout()

    _save_figure(fig, save_path)

    return fig, ax


# ---------------------------------------------------------------------
# Precision Recall Curve
# ---------------------------------------------------------------------

def plot_precision_recall_curve(
    y_true: Iterable,
    y_scores: Iterable,
    save_path: str | Path | None = None,
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Plot precision-recall curve.
    """

    y_true = _ensure_numpy(y_true)
    y_scores = _ensure_numpy(y_scores)

    precision, recall, _ = precision_recall_curve(y_true, y_scores)

    fig, ax = plt.subplots()

    ax.plot(recall, precision)

    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision–Recall Curve")

    fig.tight_layout()

    _save_figure(fig, save_path)

    return fig, ax


# ---------------------------------------------------------------------
# Calibration Curve
# ---------------------------------------------------------------------

def plot_calibration_curve(
    y_true: Iterable,
    y_prob: Iterable,
    n_bins: int = 10,
    save_path: str | Path | None = None,
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Plot calibration curve for probabilistic classifiers.
    """

    y_true = _ensure_numpy(y_true)
    y_prob = _ensure_numpy(y_prob)

    prob_true, prob_pred = calibration_curve(
        y_true,
        y_prob,
        n_bins=n_bins,
    )

    fig, ax = plt.subplots()

    ax.plot(prob_pred, prob_true, marker="o", label="Model")
    ax.plot([0, 1], [0, 1], linestyle="--", label="Perfect")

    ax.set_xlabel("Predicted Probability")
    ax.set_ylabel("True Probability")
    ax.set_title("Calibration Curve")
    ax.legend()

    fig.tight_layout()

    _save_figure(fig, save_path)

    return fig, ax


# ---------------------------------------------------------------------
# Training Curves
# ---------------------------------------------------------------------

def plot_training_curves(
    history: Dict[str, List[float]],
    save_path: str | Path | None = None,
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Plot training and validation curves.
    """

    if not history:
        raise ValueError("history dictionary cannot be empty")

    fig, ax = plt.subplots()

    for key, values in history.items():
        ax.plot(values, label=key)

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Metric")
    ax.set_title("Training Curves")
    ax.legend()

    fig.tight_layout()

    _save_figure(fig, save_path)

    return fig, ax


# ---------------------------------------------------------------------
# Feature Importance
# ---------------------------------------------------------------------

def plot_feature_importance(
    features: List[str],
    scores: Iterable,
    top_k: int = 20,
    save_path: str | Path | None = None,
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Plot feature importance ranking.
    """

    scores = _ensure_numpy(scores)

    if len(features) != len(scores):
        raise ValueError("features and scores length mismatch")

    indices = np.argsort(scores)[::-1][:top_k]

    top_features = [features[i] for i in indices]
    top_scores = scores[indices]

    fig, ax = plt.subplots(figsize=(8, 6))

    sns.barplot(
        x=top_scores,
        y=top_features,
        orient="h",
        ax=ax,
    )

    ax.set_title("Feature Importance")

    fig.tight_layout()

    _save_figure(fig, save_path)

    return fig, ax


# ---------------------------------------------------------------------
# Embedding Projection
# ---------------------------------------------------------------------

def plot_embedding_projection(
    embeddings: np.ndarray,
    labels: Optional[Iterable] = None,
    method: str = "pca",
    save_path: str | Path | None = None,
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Project high-dimensional embeddings to 2D using PCA or t-SNE.
    """

    embeddings = _ensure_numpy(embeddings)

    if embeddings.ndim != 2:
        raise ValueError("embeddings must be a 2D matrix")

    if method.lower() == "pca":
        reducer = PCA(n_components=2)

    elif method.lower() == "tsne":
        perplexity = min(30.0, max(5.0, embeddings.shape[0] - 1))
        reducer = TSNE(n_components=2, perplexity=perplexity)

    else:
        raise ValueError("method must be 'pca' or 'tsne'")

    projected = reducer.fit_transform(embeddings)

    fig, ax = plt.subplots()

    if labels is None:

        ax.scatter(
            projected[:, 0],
            projected[:, 1],
            alpha=0.7,
        )

    else:

        labels = _ensure_numpy(labels)

        for label in np.unique(labels):

            idx = labels == label

            ax.scatter(
                projected[idx, 0],
                projected[idx, 1],
                label=str(label),
                alpha=0.7,
            )

        ax.legend()

    ax.set_title(f"Embedding Projection ({method.upper()})")

    fig.tight_layout()

    _save_figure(fig, save_path)

    return fig, ax