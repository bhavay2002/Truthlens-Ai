from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Literal, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


# =========================================================
# CONFIG
# =========================================================

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


# =========================================================
# FACTORY
# =========================================================

class LossFactory:

    SUPPORTED = {
        "binary",
        "multi_class",
        "multi_label",
        "regression",
    }

    @staticmethod
    def create(config: LossConfig) -> nn.Module:

        if config.loss_type not in LossFactory.SUPPORTED:
            raise ValueError(f"Unsupported loss: {config.loss_type}")

        if config.reduction not in {"mean", "sum", "none"}:
            raise ValueError("Invalid reduction")

        if not (0.0 <= config.label_smoothing < 1.0):
            raise ValueError("Invalid label_smoothing")

        weight = float(config.weight)

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
            raise RuntimeError

        if weight != 1.0:
            return WeightedLoss(base, weight)

        return base


# =========================================================
# WRAPPER
# =========================================================

class WeightedLoss(nn.Module):

    def __init__(self, base: nn.Module, weight: float):
        super().__init__()
        self.base = base
        self.weight = weight

    def forward(self, logits: torch.Tensor, targets: torch.Tensor):
        return self.base(logits, targets) * self.weight


# =========================================================
# TASK TYPES
# =========================================================

MULTICLASS_TASKS = {"bias", "ideology", "propaganda"}
MULTILABEL_TASKS = {"narrative", "narrative_frame", "emotion"}


def get_task_type(task: str) -> str:
    if task in MULTICLASS_TASKS:
        return "multiclass"
    if task in MULTILABEL_TASKS:
        return "multilabel"
    raise ValueError(f"Unknown task: {task}")


# =========================================================
# LOSSES
# =========================================================

def binary_loss(logits: torch.Tensor, targets: torch.Tensor):

    targets = targets.float()

    if targets.dim() == 1:
        targets = targets.unsqueeze(1)

    if logits.shape != targets.shape:
        raise RuntimeError("Shape mismatch")

    return F.binary_cross_entropy_with_logits(logits.float(), targets)


def multiclass_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    ignore_index: int = -100,
):

    if targets.dim() == 2:
        targets = targets.argmax(dim=1)

    mask = targets != ignore_index

    if not mask.any():
        return logits.sum() * 0.0

    return F.cross_entropy(
        logits[mask].float(),
        targets[mask].long(),
    )


def multilabel_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    ignore_index: float = -100.0,
):

    targets = targets.float()

    if logits.shape != targets.shape:
        raise RuntimeError("Shape mismatch")

    mask = targets != ignore_index

    if not mask.any():
        return logits.sum() * 0.0

    safe_targets = torch.where(mask, targets, torch.zeros_like(targets))

    loss = F.binary_cross_entropy_with_logits(
        logits.float(),
        safe_targets,
        reduction="none",
    )

    loss = loss * mask.float()

    return loss.sum() / mask.sum().clamp_min(1)


def regression_loss(
    preds: torch.Tensor,
    targets: torch.Tensor,
):

    if preds.shape != targets.shape:
        raise RuntimeError("Shape mismatch")

    return F.mse_loss(preds.float(), targets.float())


# =========================================================
# MULTI-TASK LOSS
# =========================================================

def compute_task_loss(
    task: str,
    logits: torch.Tensor,
    labels: torch.Tensor,
) -> torch.Tensor:

    if task in MULTICLASS_TASKS:
        return multiclass_loss(logits, labels)

    if task in MULTILABEL_TASKS:
        return multilabel_loss(logits, labels)

    raise ValueError(f"Unknown task: {task}")


def compute_multitask_loss(
    outputs: Dict[str, torch.Tensor],
    labels: Dict[str, torch.Tensor],
    weights: Dict[str, float] | None = None,
) -> torch.Tensor:

    total_loss = 0.0

    for task, logits in outputs.items():

        if task not in labels:
            continue

        loss = compute_task_loss(task, logits, labels[task])

        if weights and task in weights:
            loss = loss * weights[task]

        total_loss = total_loss + loss

    return total_loss