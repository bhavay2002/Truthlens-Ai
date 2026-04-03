"""
File Name: uncertainty.py
Module: TruthLens AI - Uncertainty Metrics
Description:
    Utilities for estimating predictive uncertainty from probabilistic model
    outputs. Provides entropy-based uncertainty, confidence estimation,
    and aggregated uncertainty statistics used in TruthLens evaluation and
    diagnostics pipelines.
Dependencies:
    numpy
    logging
    typing
Inputs:
    probs: Model probability outputs (N x C) where C is number of classes
Outputs:
    Entropy values, confidence scores, and aggregated uncertainty statistics
"""

from __future__ import annotations

import logging
from typing import Dict, Iterable

import numpy as np


logger = logging.getLogger(__name__)


EPS = 1e-12


def _validate_probs(probs: Iterable) -> np.ndarray:
    """
    Validate probability tensor.
    """

    probs_arr = np.asarray(probs, dtype=float)

    if probs_arr.ndim != 2:
        raise ValueError(
            "Probability array must be 2D with shape (n_samples, n_classes)."
        )

    if probs_arr.shape[0] == 0:
        raise ValueError("Probability array cannot be empty.")

    if np.any(probs_arr < 0) or np.any(probs_arr > 1):
        logger.warning("Probability values outside [0,1] detected.")

    row_sums = probs_arr.sum(axis=1)

    if not np.allclose(row_sums, 1.0, atol=1e-3):
        logger.warning(
            "Probabilities do not sum to 1 across classes. "
            "Results may be unreliable."
        )

    return probs_arr


def predictive_entropy(probs: Iterable) -> np.ndarray:
    """
    Compute predictive entropy for each sample.
    """

    probs_arr = _validate_probs(probs)

    entropy = -np.sum(
        probs_arr * np.log(probs_arr + EPS),
        axis=1
    )

    logger.debug("Computed predictive entropy for %d samples", entropy.shape[0])

    return entropy


def confidence_scores(probs: Iterable) -> np.ndarray:
    """
    Compute model confidence scores as max probability per sample.
    """

    probs_arr = _validate_probs(probs)

    confidence = np.max(probs_arr, axis=1)

    logger.debug(
        "Computed confidence scores for %d samples",
        confidence.shape[0]
    )

    return confidence


def uncertainty_statistics(probs: Iterable) -> Dict[str, float]:
    """
    Compute aggregated uncertainty statistics.
    """

    probs_arr = _validate_probs(probs)

    entropy = predictive_entropy(probs_arr)
    confidence = confidence_scores(probs_arr)

    stats: Dict[str, float] = {
        "mean_entropy": float(np.mean(entropy)),
        "std_entropy": float(np.std(entropy)),
        "max_entropy": float(np.max(entropy)),
        "min_entropy": float(np.min(entropy)),
        "mean_confidence": float(np.mean(confidence)),
        "std_confidence": float(np.std(confidence)),
        "max_confidence": float(np.max(confidence)),
        "min_confidence": float(np.min(confidence)),
    }

    logger.info("Computed uncertainty statistics")

    return stats
