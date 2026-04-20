"""
File Name: calibration.py
Module: TruthLens AI - Calibration Analysis
Description:
    Calibration analysis utilities for evaluating probabilistic predictions
    produced by TruthLens AI models. Implements Expected Calibration Error (ECE)
    computation and reliability diagram generation. Designed for research-grade
    evaluation of model confidence calibration.
Dependencies:
    numpy
    matplotlib
    sklearn.calibration
    logging
    pathlib
    typing
Inputs:
    y_true: Ground truth binary labels
    probs: Predicted probabilities for the positive class
    n_bins: Number of calibration bins
    save_path: File path to store reliability diagram
Outputs:
    Expected Calibration Error (float)
    Reliability diagram image file
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable, Tuple

import numpy as np
import matplotlib.pyplot as plt
import torch
from src.models.calibration import CalibrationMetricConfig, CalibrationMetrics


logger = logging.getLogger(__name__)


def _validate_inputs(
    y_true: Iterable,
    probs: Iterable
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Validate and convert calibration inputs.
    """

    y_true_arr = np.asarray(y_true)
    probs_arr = np.asarray(probs)

    if y_true_arr.shape[0] == 0:
        raise ValueError("y_true cannot be empty.")

    if y_true_arr.shape[0] != probs_arr.shape[0]:
        raise ValueError("y_true and probs must have the same length.")

    if np.any(probs_arr < 0) or np.any(probs_arr > 1):
        raise ValueError("Probabilities must be within [0, 1].")

    labels = y_true_arr.astype(np.int64)
    if labels.min() < 0:
        raise ValueError("Labels must be non-negative integers for calibration.")

    return y_true_arr, probs_arr


def expected_calibration_error(
    y_true: Iterable,
    probs: Iterable,
    n_bins: int = 10
) -> float:
    """
    Compute Expected Calibration Error (ECE).

    ECE measures the difference between model confidence and accuracy
    across probability bins.
    """

    y_true_arr, probs_arr = _validate_inputs(y_true, probs)
    metric = CalibrationMetrics(CalibrationMetricConfig(n_bins=n_bins))

    if probs_arr.ndim == 1:
        probs_2d = np.stack([1.0 - probs_arr, probs_arr], axis=1)
    else:
        probs_2d = probs_arr

    labels = y_true_arr.astype(np.int64)
    if labels.min() < 0:
        raise ValueError("Labels must be non-negative integers for calibration.")

    ece = metric.expected_calibration_error(
        torch.tensor(probs_2d, dtype=torch.float32),
        torch.tensor(labels, dtype=torch.long),
    )

    logger.info("Expected Calibration Error (ECE): %.6f", ece)

    return ece


def plot_reliability_diagram(
    y_true: Iterable,
    probs: Iterable,
    save_path: str | Path,
    n_bins: int = 10
) -> Path:
    """
    Generate and save reliability diagram for model calibration.
    """

    y_true_arr, probs_arr = _validate_inputs(y_true, probs)

    metric = CalibrationMetrics(CalibrationMetricConfig(n_bins=n_bins))

    if probs_arr.ndim == 1:
        probs_2d = np.stack([1.0 - probs_arr, probs_arr], axis=1)
    else:
        probs_2d = probs_arr

    labels = y_true_arr.astype(np.int64)
    stats = metric.reliability_statistics(
        torch.tensor(probs_2d, dtype=torch.float32),
        torch.tensor(labels, dtype=torch.long),
    )
    prob_true = stats["bin_accuracy"]
    prob_pred = stats["bin_confidence"]

    output_path = Path(save_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(6, 6))

    plt.plot(prob_pred, prob_true, marker="o", label="Model")
    plt.plot([0, 1], [0, 1], linestyle="--", label="Perfect Calibration")

    plt.xlabel("Confidence")
    plt.ylabel("Accuracy")
    plt.title("Reliability Diagram")
    plt.legend()

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

    logger.info("Reliability diagram saved to %s", output_path)

    return output_path
