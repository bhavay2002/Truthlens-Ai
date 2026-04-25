from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, Any

import numpy as np
import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)


# =========================================================
# CONFIG
# =========================================================

@dataclass
class CalibrationMetricConfig:
    n_bins: int = 15

    def __post_init__(self):
        if self.n_bins <= 1:
            raise ValueError("n_bins must be > 1")


# =========================================================
# METRICS
# =========================================================

class CalibrationMetrics:

    def __init__(self, config: CalibrationMetricConfig | None = None):
        self.config = config or CalibrationMetricConfig()

    # -----------------------------------------------------

    @staticmethod
    def _validate(probs: torch.Tensor, labels: torch.Tensor):
        if probs.ndim != 2:
            raise ValueError("probs must be [N, C]")
        if labels.ndim != 1:
            raise ValueError("labels must be [N]")
        if probs.shape[0] != labels.shape[0]:
            raise ValueError("size mismatch")

    # -----------------------------------------------------

    @staticmethod
    def _to_probs(x: torch.Tensor) -> torch.Tensor:
        if x.max() > 1 or x.min() < 0:
            return torch.softmax(x, dim=1)
        return x

    # =====================================================
    # ECE
    # =====================================================

    def expected_calibration_error(
        self,
        logits_or_probs: torch.Tensor,
        labels: torch.Tensor,
    ) -> float:

        probs = self._to_probs(logits_or_probs)
        self._validate(probs, labels)

        conf, preds = torch.max(probs, dim=1)
        acc = preds.eq(labels)

        bins = torch.linspace(0, 1, self.config.n_bins + 1)
        ece = torch.zeros(1, device=probs.device)

        for i in range(self.config.n_bins):
            mask = (conf > bins[i]) & (conf <= bins[i + 1])

            if mask.sum() > 0:
                bin_acc = acc[mask].float().mean()
                bin_conf = conf[mask].mean()
                weight = mask.float().mean()
                ece += torch.abs(bin_conf - bin_acc) * weight

        return float(ece.item())

    # =====================================================
    # MCE
    # =====================================================

    def maximum_calibration_error(
        self,
        logits_or_probs: torch.Tensor,
        labels: torch.Tensor,
    ) -> float:

        probs = self._to_probs(logits_or_probs)
        self._validate(probs, labels)

        conf, preds = torch.max(probs, dim=1)
        acc = preds.eq(labels)

        bins = torch.linspace(0, 1, self.config.n_bins + 1)
        mce = torch.zeros(1, device=probs.device)

        for i in range(self.config.n_bins):
            mask = (conf > bins[i]) & (conf <= bins[i + 1])

            if mask.sum() > 0:
                error = torch.abs(
                    conf[mask].mean() - acc[mask].float().mean()
                )
                mce = torch.maximum(mce, error)

        return float(mce.item())

    # =====================================================
    # BRIER
    # =====================================================

    def brier_score(
        self,
        logits_or_probs: torch.Tensor,
        labels: torch.Tensor,
    ) -> float:

        probs = self._to_probs(logits_or_probs)
        self._validate(probs, labels)

        one_hot = F.one_hot(labels, num_classes=probs.shape[1]).float()
        score = torch.mean(torch.sum((probs - one_hot) ** 2, dim=1))

        return float(score.item())

    # =====================================================
    # NLL
    # =====================================================

    def negative_log_likelihood(
        self,
        logits_or_probs: torch.Tensor,
        labels: torch.Tensor,
    ) -> float:

        if logits_or_probs.max() <= 1 and logits_or_probs.min() >= 0:
            probs = logits_or_probs
            loss = F.nll_loss(torch.log(probs + 1e-12), labels)
        else:
            loss = F.cross_entropy(logits_or_probs, labels)

        return float(loss.item())

    # =====================================================
    # RELIABILITY
    # =====================================================

    def reliability_statistics(
        self,
        logits_or_probs: torch.Tensor,
        labels: torch.Tensor,
    ) -> Dict[str, np.ndarray]:

        probs = self._to_probs(logits_or_probs)
        self._validate(probs, labels)

        conf, preds = torch.max(probs, dim=1)
        acc = preds.eq(labels).float()

        conf_np = conf.cpu().numpy()
        acc_np = acc.cpu().numpy()

        bins = np.linspace(0.0, 1.0, self.config.n_bins + 1)

        bin_acc = np.zeros(self.config.n_bins)
        bin_conf = np.zeros(self.config.n_bins)
        bin_counts = np.zeros(self.config.n_bins)

        for i in range(self.config.n_bins):
            mask = (conf_np > bins[i]) & (conf_np <= bins[i + 1])

            if mask.sum() > 0:
                bin_acc[i] = acc_np[mask].mean()
                bin_conf[i] = conf_np[mask].mean()
                bin_counts[i] = mask.sum()

        return {
            "bin_accuracy": bin_acc,
            "bin_confidence": bin_conf,
            "bin_counts": bin_counts,
            "bin_boundaries": bins,
        }

    # =====================================================
    # ALL
    # =====================================================

    def compute_all_metrics(
        self,
        logits_or_probs: torch.Tensor,
        labels: torch.Tensor,
    ) -> Dict[str, float]:

        metrics = {
            "ece": self.expected_calibration_error(logits_or_probs, labels),
            "mce": self.maximum_calibration_error(logits_or_probs, labels),
            "brier_score": self.brier_score(logits_or_probs, labels),
            "nll": self.negative_log_likelihood(logits_or_probs, labels),
        }

        logger.info("Calibration metrics: %s", metrics)

        return metrics