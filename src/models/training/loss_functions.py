"""
File Name: loss_functions.py
Module: models.training
Description:
    Provides reusable loss function utilities used throughout the TruthLens
    training pipeline. The module centralizes loss definitions for common
    machine learning objectives including:

        - binary classification
        - multi-class classification
        - multi-label classification
        - regression

    It also provides a configurable loss factory and optional task weighting
    mechanisms for multi-task training setups.

    The goal is to avoid scattered loss definitions across training code and
    ensure consistent, validated loss computation across the system.
    
Dependencies:
    logging
    typing
    dataclasses
    torch
    torch.nn
Inputs:
    Model logits and corresponding labels
Outputs:
    Computed loss tensor
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


@dataclass
class LossConfig:
    """
    Configuration describing a loss function.
    """

    loss_type: Literal[
        "binary",
        "multi_class",
        "multi_label",
        "regression",
    ]

    weight: float = 1.0
    label_smoothing: float = 0.0
    reduction: Literal["mean", "sum", "none"] = "mean"


class LossFactory:
    """
    Factory responsible for creating loss functions.
    """

    SUPPORTED_LOSSES = {
        "binary",
        "multi_class",
        "multi_label",
        "regression",
    }

    @staticmethod
    def create(config: LossConfig) -> nn.Module:

        if config.loss_type not in LossFactory.SUPPORTED_LOSSES:
            raise ValueError(
                f"Unsupported loss_type '{config.loss_type}'. "
                f"Supported: {LossFactory.SUPPORTED_LOSSES}"
            )

        if config.reduction not in {"mean", "sum", "none"}:
            raise ValueError("Invalid reduction value")

        if not 0.0 <= config.label_smoothing < 1.0:
            raise ValueError("label_smoothing must be in [0,1)")

        if config.loss_type == "binary":

            base = nn.BCEWithLogitsLoss(reduction=config.reduction)

        elif config.loss_type == "multi_class":

            base = nn.CrossEntropyLoss(
                label_smoothing=config.label_smoothing,
                reduction=config.reduction,
            )

        elif config.loss_type == "multi_label":

            base = nn.BCEWithLogitsLoss(reduction=config.reduction)

        elif config.loss_type == "regression":

            base = nn.MSELoss(reduction=config.reduction)

        else:
            raise RuntimeError("Unexpected loss type")

        if config.weight != 1.0:
            base = WeightedLossWrapper(base, config.weight)

        return base


class WeightedLossWrapper(nn.Module):
    """
    Applies scalar weight to a base loss function.
    """

    def __init__(self, base_loss: nn.Module, weight: float = 1.0) -> None:
        super().__init__()

        if not isinstance(base_loss, nn.Module):
            raise TypeError("base_loss must be nn.Module")

        if weight <= 0:
            raise ValueError("weight must be positive")

        self.base_loss = base_loss
        self.weight = weight

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:

        loss = self.base_loss(logits, targets)

        return loss * self.weight


# ---------------------------------------------------------
# Functional Helper Losses
# ---------------------------------------------------------

def binary_classification_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
) -> torch.Tensor:

    targets = targets.float()

    if targets.dim() == 1:
        targets = targets.unsqueeze(1)

    if logits.shape != targets.shape:
        raise RuntimeError(
            f"Binary loss shape mismatch: logits {logits.shape} vs targets {targets.shape}"
        )

    return F.binary_cross_entropy_with_logits(logits, targets)


def multiclass_classification_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
) -> torch.Tensor:

    if targets.dim() == 2:
        targets = targets.argmax(dim=1)

    if logits.dim() != 2:
        raise RuntimeError("Multi-class logits must be 2D")

    return F.cross_entropy(logits, targets.long())


def multilabel_classification_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
) -> torch.Tensor:

    targets = targets.float()

    if logits.shape != targets.shape:
        raise RuntimeError(
            f"Multi-label shape mismatch: logits {logits.shape} vs targets {targets.shape}"
        )

    return F.binary_cross_entropy_with_logits(logits, targets)


def regression_loss(
    predictions: torch.Tensor,
    targets: torch.Tensor,
) -> torch.Tensor:

    targets = targets.float()

    if predictions.shape != targets.shape:
        raise RuntimeError(
            f"Regression shape mismatch: predictions {predictions.shape} vs targets {targets.shape}"
        )

    return F.mse_loss(predictions, targets)