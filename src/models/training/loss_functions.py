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


# ---------------------------------------------------------
# Config
# ---------------------------------------------------------

@dataclass
class LossConfig:
    loss_type: Literal[
        "binary",
        "multi_class",
        "multi_label",
        "regression",
    ]
    weight: float = 1.0
    label_smoothing: float = 0.0
    reduction: Literal["mean", "sum", "none"] = "mean"


# ---------------------------------------------------------
# Factory
# ---------------------------------------------------------

class LossFactory:
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

        if not isinstance(config.weight, (int, float)):
            raise TypeError("weight must be numeric")

        weight = float(config.weight)

        if config.loss_type == "binary":
            base = nn.BCEWithLogitsLoss(reduction=config.reduction)
            if weight != 1.0:
                return lambda logits, targets: base(logits, targets) * weight
            return base

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

        if weight != 1.0:
            base = WeightedLossWrapper(base, weight)

        return base


# ---------------------------------------------------------
# Wrapper
# ---------------------------------------------------------

class WeightedLossWrapper(nn.Module):
    def __init__(self, base_loss: nn.Module, weight: float = 1.0) -> None:
        super().__init__()

        if not isinstance(base_loss, nn.Module):
            raise TypeError("base_loss must be nn.Module")

        if weight <= 0:
            raise ValueError("weight must be positive")

        self.base_loss = base_loss
        self.weight = weight

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # compile-friendly + avoids extra ops
        return self.base_loss(logits, targets).mul(self.weight)


# ---------------------------------------------------------
# Functional Helper Losses (AMP-safe)
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

    return F.binary_cross_entropy_with_logits(
        logits.float(),  # AMP safety
        targets
    )


MULTICLASS_TASKS: frozenset = frozenset({"bias", "ideology", "propaganda"})
MULTILABEL_TASKS: frozenset = frozenset({"narrative", "narrative_frame", "emotion"})


def get_task_type(task: str) -> str:
    """Return 'multiclass' or 'multilabel' for a known task name."""
    if task in MULTICLASS_TASKS:
        return "multiclass"
    if task in MULTILABEL_TASKS:
        return "multilabel"
    raise ValueError(f"Unknown task: {task!r}")


def multiclass_classification_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    ignore_index: int = -100,
) -> torch.Tensor:
    """Cross-entropy loss with -100 masking support."""

    if targets.dim() == 2:
        targets = targets.argmax(dim=1)

    if __debug__:
        if logits.dim() != 2:
            raise RuntimeError("Multi-class logits must be 2D")

    valid_mask = targets.ne(ignore_index)
    if not valid_mask.any():
        return logits.sum() * 0.0

    return F.cross_entropy(
        logits[valid_mask].float(),
        targets[valid_mask].long(),
    )


def multilabel_classification_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    ignore_index: float = -100.0,
) -> torch.Tensor:
    """BCE loss with per-element -100 masking support."""

    targets = targets.float()

    if __debug__:
        if logits.shape != targets.shape:
            raise RuntimeError(
                f"Multi-label shape mismatch: logits {logits.shape} vs targets {targets.shape}"
            )

    valid_mask = targets.ne(ignore_index)
    if not valid_mask.any():
        return logits.sum() * 0.0

    safe_targets = torch.where(valid_mask, targets, torch.zeros_like(targets))
    raw_loss = F.binary_cross_entropy_with_logits(
        logits.float(),
        safe_targets,
        reduction="none",
    )
    masked_loss = raw_loss * valid_mask.float()
    return masked_loss.sum() / valid_mask.sum().clamp_min(1)


def compute_task_loss(
    task: str,
    logits: torch.Tensor,
    labels: torch.Tensor,
) -> torch.Tensor:
    """Unified loss router — picks the correct loss function for a task.

    Parameters
    ----------
    task:
        One of the six TruthLens task names.
    logits:
        Raw model logits (not softmaxed/sigmoided).
    labels:
        Ground-truth labels.  -100 entries are masked out automatically.
    """
    if task in MULTICLASS_TASKS:
        return multiclass_classification_loss(logits, labels)
    if task in MULTILABEL_TASKS:
        return multilabel_classification_loss(logits, labels)
    raise ValueError(f"Unknown task: {task!r}")


def regression_loss(
    predictions: torch.Tensor,
    targets: torch.Tensor,
) -> torch.Tensor:

    targets = targets.float()

    if __debug__:
        if predictions.shape != targets.shape:
            raise RuntimeError(
                f"Regression shape mismatch: predictions {predictions.shape} vs targets {targets.shape}"
            )

    return F.mse_loss(
        predictions.float(),  # AMP safety
        targets
    )