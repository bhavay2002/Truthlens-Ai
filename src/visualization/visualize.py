"""
File: visualise.py

Purpose
-------
Visualization utilities for TruthLens AI.

This module provides plotting functions used for model evaluation,
including confusion matrices and performance analysis charts.
"""

import logging
from typing import Tuple

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


# ---------------------------------------------------------
# Logging
# ---------------------------------------------------------

logger = logging.getLogger(__name__)


# ---------------------------------------------------------
# Confusion Matrix Plot
# ---------------------------------------------------------

def plot_confusion_matrix(
    cm: np.ndarray,
    labels: list[str] | None = None,
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Plot confusion matrix heatmap.

    Parameters
    ----------
    cm : np.ndarray
        Confusion matrix.

    labels : list[str], optional
        Class labels.

    Returns
    -------
    fig, ax
    """

    try:

        logger.info("Generating confusion matrix visualization")

        cm = np.asarray(cm)
        if cm.ndim != 2:
            raise ValueError("cm must be a 2D array")
        if cm.shape[0] != cm.shape[1]:
            raise ValueError("cm must be a square matrix")

        if labels is None:
            labels = ["REAL", "FAKE"]
        if len(labels) != cm.shape[0]:
            raise ValueError(
                "labels length must match confusion matrix dimensions "
                f"({len(labels)} != {cm.shape[0]})"
            )

        fig, ax = plt.subplots(figsize=(6, 5))

        sns.heatmap(
            cm,
            annot=True,
            fmt="g",
            cmap="Blues",
            cbar=False,
            xticklabels=labels,
            yticklabels=labels,
            ax=ax,
        )

        ax.set_xlabel("Predicted Label")
        ax.set_ylabel("True Label")
        ax.set_title("Confusion Matrix")

        fig.tight_layout()

        return fig, ax

    except Exception:

        logger.exception("Failed to generate confusion matrix plot")

        raise
