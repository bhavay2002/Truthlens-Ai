"""
File Name: calibration_metrics.py
Module: evaluation.calibration
Description:
    Implements calibration evaluation metrics used to assess the reliability
    of probabilistic predictions produced by classification models.

    The module includes commonly used calibration metrics such as:

    • Expected Calibration Error (ECE)
    • Maximum Calibration Error (MCE)
    • Brier Score
    • Negative Log Likelihood (NLL)
    • Reliability diagram statistics

    These metrics are essential for evaluating whether predicted probabilities
    accurately reflect true likelihoods. The implementation supports both
    binary and multiclass classification settings and integrates with PyTorch
    pipelines by accepting logits or probability tensors.

    Designed for research-grade and production ML systems with robust input
    validation, structured logging, and reproducibility.
    
Dependencies:
    numpy
    torch
    logging
    dataclasses
    typing
Inputs:
    Model logits or probability predictions and ground truth labels.
Outputs:
    Calibration metric values and bin statistics.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict

import numpy as np
import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)


@dataclass
class CalibrationMetricConfig:
    """
    Configuration for calibration metric computation.
    """

    n_bins: int = 15

    def __post_init__(self) -> None:
        if self.n_bins <= 1:
            raise ValueError("n_bins must be greater than 1.")


class CalibrationMetrics:
    """
    Computes calibration metrics for classification models.

    Supported metrics:
        • Expected Calibration Error (ECE)
        • Maximum Calibration Error (MCE)
        • Brier Score
        • Negative Log Likelihood (NLL)
    """

    def __init__(self, config: CalibrationMetricConfig | None = None) -> None:
        self.config = config or CalibrationMetricConfig()

    @staticmethod
    def _validate_inputs(probs: torch.Tensor, labels: torch.Tensor) -> None:
        """Validate shapes and types of inputs."""
        if probs.ndim != 2:
            raise ValueError(
                "Probabilities must have shape [num_samples, num_classes]."
            )

        if labels.ndim != 1:
            raise ValueError("Labels must be a 1D tensor.")

        if probs.shape[0] != labels.shape[0]:
            raise ValueError(
                "Number of predictions must match number of labels."
            )

    @staticmethod
    def _logits_to_probs(logits: torch.Tensor) -> torch.Tensor:
        """Convert logits to probabilities using softmax."""
        return torch.softmax(logits, dim=1)

    def _prepare_probs(
        self, logits_or_probs: torch.Tensor
    ) -> torch.Tensor:
        """Ensure predictions are probabilities."""
        if logits_or_probs.max() > 1 or logits_or_probs.min() < 0:
            return self._logits_to_probs(logits_or_probs)
        return logits_or_probs

    def expected_calibration_error(
        self,
        logits_or_probs: torch.Tensor,
        labels: torch.Tensor,
    ) -> float:
        """
        Compute Expected Calibration Error (ECE).
        """

        probs = self._prepare_probs(logits_or_probs)
        self._validate_inputs(probs, labels)

        confidences, predictions = torch.max(probs, dim=1)
        accuracies = predictions.eq(labels)

        n_bins = self.config.n_bins
        bin_boundaries = torch.linspace(0, 1, n_bins + 1)

        ece = torch.zeros(1, device=probs.device)

        for i in range(n_bins):
            bin_lower = bin_boundaries[i]
            bin_upper = bin_boundaries[i + 1]

            mask = (confidences > bin_lower) * (confidences <= bin_upper)

            if mask.sum() > 0:
                accuracy = accuracies[mask].float().mean()
                confidence = confidences[mask].mean()

                bin_prob = mask.float().mean()

                ece += torch.abs(confidence - accuracy) * bin_prob

        return float(ece.item())

    def maximum_calibration_error(
        self,
        logits_or_probs: torch.Tensor,
        labels: torch.Tensor,
    ) -> float:
        """
        Compute Maximum Calibration Error (MCE).
        """

        probs = self._prepare_probs(logits_or_probs)
        self._validate_inputs(probs, labels)

        confidences, predictions = torch.max(probs, dim=1)
        accuracies = predictions.eq(labels)

        n_bins = self.config.n_bins
        bin_boundaries = torch.linspace(0, 1, n_bins + 1)

        mce = torch.zeros(1, device=probs.device)

        for i in range(n_bins):
            bin_lower = bin_boundaries[i]
            bin_upper = bin_boundaries[i + 1]

            mask = (confidences > bin_lower) * (confidences <= bin_upper)

            if mask.sum() > 0:
                accuracy = accuracies[mask].float().mean()
                confidence = confidences[mask].mean()

                error = torch.abs(confidence - accuracy)

                mce = torch.maximum(mce, error)

        return float(mce.item())

    def brier_score(
        self,
        logits_or_probs: torch.Tensor,
        labels: torch.Tensor,
    ) -> float:
        """
        Compute Brier score.
        """

        probs = self._prepare_probs(logits_or_probs)
        self._validate_inputs(probs, labels)

        num_classes = probs.shape[1]

        one_hot = F.one_hot(labels, num_classes=num_classes).float()

        score = torch.mean(torch.sum((probs - one_hot) ** 2, dim=1))

        return float(score.item())

    def negative_log_likelihood(
        self,
        logits_or_probs: torch.Tensor,
        labels: torch.Tensor,
    ) -> float:
        """
        Compute Negative Log Likelihood (NLL).
        """

        if logits_or_probs.max() <= 1 and logits_or_probs.min() >= 0:
            probs = logits_or_probs
            log_probs = torch.log(probs + 1e-12)
            loss = F.nll_loss(log_probs, labels)
        else:
            loss = F.cross_entropy(logits_or_probs, labels)

        return float(loss.item())

    def reliability_statistics(
        self,
        logits_or_probs: torch.Tensor,
        labels: torch.Tensor,
    ) -> Dict[str, np.ndarray]:
        """
        Compute statistics required for reliability diagrams.
        """

        probs = self._prepare_probs(logits_or_probs)
        self._validate_inputs(probs, labels)

        confidences, predictions = torch.max(probs, dim=1)
        accuracies = predictions.eq(labels).float()

        confidences_np = confidences.detach().cpu().numpy()
        accuracies_np = accuracies.detach().cpu().numpy()

        n_bins = self.config.n_bins
        bins = np.linspace(0.0, 1.0, n_bins + 1)

        bin_acc = np.zeros(n_bins)
        bin_conf = np.zeros(n_bins)
        bin_counts = np.zeros(n_bins)

        for i in range(n_bins):
            lower = bins[i]
            upper = bins[i + 1]

            mask = (confidences_np > lower) & (confidences_np <= upper)

            if mask.sum() > 0:
                bin_acc[i] = accuracies_np[mask].mean()
                bin_conf[i] = confidences_np[mask].mean()
                bin_counts[i] = mask.sum()

        return {
            "bin_accuracy": bin_acc,
            "bin_confidence": bin_conf,
            "bin_counts": bin_counts,
            "bin_boundaries": bins,
        }

    def compute_all_metrics(
        self,
        logits_or_probs: torch.Tensor,
        labels: torch.Tensor,
    ) -> Dict[str, float]:
        """
        Compute all calibration metrics.
        """

        metrics = {
            "ece": self.expected_calibration_error(logits_or_probs, labels),
            "mce": self.maximum_calibration_error(logits_or_probs, labels),
            "brier_score": self.brier_score(logits_or_probs, labels),
            "nll": self.negative_log_likelihood(logits_or_probs, labels),
        }

        logger.info("Calibration metrics computed: %s", metrics)

        return metrics
