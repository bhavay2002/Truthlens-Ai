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

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


@dataclass
class LossConfig:
    """
    Configuration describing a loss function.
    """

    loss_type: str
    weight: float = 1.0
    label_smoothing: float = 0.0
    reduction: str = "mean"


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
        """
        Create a PyTorch loss module from configuration.
        """

        if config.loss_type not in LossFactory.SUPPORTED_LOSSES:
            raise ValueError(
                f"Unsupported loss_type '{config.loss_type}'. "
                f"Supported: {LossFactory.SUPPORTED_LOSSES}"
            )

        if config.loss_type == "binary":
            return nn.BCEWithLogitsLoss(reduction=config.reduction)

        if config.loss_type == "multi_class":
            return nn.CrossEntropyLoss(
                label_smoothing=config.label_smoothing,
                reduction=config.reduction,
            )

        if config.loss_type == "multi_label":
            return nn.BCEWithLogitsLoss(reduction=config.reduction)

        if config.loss_type == "regression":
            return nn.MSELoss(reduction=config.reduction)

        raise RuntimeError("Loss creation failed unexpectedly")


class WeightedLossWrapper(nn.Module):
    """
    Applies a scalar weight to an existing loss function.

    Useful for balancing multi-task training objectives.
    """

    def __init__(self, base_loss: nn.Module, weight: float = 1.0) -> None:
        super().__init__()

        if not isinstance(base_loss, nn.Module):
            raise TypeError("base_loss must be torch.nn.Module")

        if weight <= 0:
            raise ValueError("weight must be positive")

        self.base_loss = base_loss
        self.weight = weight

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute weighted loss.
        """

        loss = self.base_loss(logits, targets)

        return loss * self.weight


def binary_classification_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
) -> torch.Tensor:
    """
    Binary classification loss helper.
    """

    if targets.dim() == 1:
        targets = targets.unsqueeze(1)

    loss_fn = nn.BCEWithLogitsLoss()

    return loss_fn(logits, targets.float())


def multiclass_classification_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
) -> torch.Tensor:
    """
    Multi-class classification loss helper.
    """

    if targets.dim() == 2:
        targets = targets.argmax(dim=1)

    loss_fn = nn.CrossEntropyLoss()

    return loss_fn(logits, targets.long())


def multilabel_classification_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
) -> torch.Tensor:
    """
    Multi-label classification loss helper.
    """

    loss_fn = nn.BCEWithLogitsLoss()

    return loss_fn(logits, targets.float())


def regression_loss(
    predictions: torch.Tensor,
    targets: torch.Tensor,
) -> torch.Tensor:
    """
    Regression loss helper.
    """

    loss_fn = nn.MSELoss()

    return loss_fn(predictions, targets.float())
