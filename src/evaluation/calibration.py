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
from sklearn.calibration import calibration_curve


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

    bins = np.linspace(0.0, 1.0, n_bins + 1)
    bin_ids = np.digitize(probs_arr, bins) - 1

    ece = 0.0
    total = len(probs_arr)

    for i in range(n_bins):

        mask = bin_ids == i
        bin_count = np.sum(mask)

        if bin_count == 0:
            continue

        acc = np.mean(y_true_arr[mask])
        conf = np.mean(probs_arr[mask])

        ece += np.abs(acc - conf) * (bin_count / total)

    ece = float(ece)

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

    prob_true, prob_pred = calibration_curve(
        y_true_arr,
        probs_arr,
        n_bins=n_bins
    )

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