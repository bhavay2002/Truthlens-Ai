#src\models\training\loss_functions.py
from __future__ import annotations

import torch
import torch.nn.functional as F


# =========================================================
# PURE LOSS FUNCTIONS (NO TASK LOGIC)
# =========================================================

def binary_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    targets = targets.float()

    if targets.dim() == 1:
        targets = targets.unsqueeze(1)

    if logits.shape != targets.shape:
        raise RuntimeError("Shape mismatch in binary_loss")

    return F.binary_cross_entropy_with_logits(logits.float(), targets)


def multiclass_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    ignore_index: int = -100,
) -> torch.Tensor:

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
) -> torch.Tensor:

    targets = targets.float()

    if logits.shape != targets.shape:
        raise RuntimeError("Shape mismatch in multilabel_loss")

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
) -> torch.Tensor:

    if preds.shape != targets.shape:
        raise RuntimeError("Shape mismatch in regression_loss")

    return F.mse_loss(preds.float(), targets.float())

# =========================================================
# COMPAT: minimal Loss config + factory stubs
# =========================================================

from dataclasses import dataclass as _dataclass


@_dataclass
class LossConfig:
    """Lightweight loss-config used by classifier modules."""
    task_type: str = "multiclass"
    label_smoothing: float = 0.0
    pos_weight: float | None = None


class LossFactory:
    """Tiny dispatcher used by classifier modules.

    Returns one of the loss functions defined above based on ``config.task_type``.
    """

    @staticmethod
    def create(config: "LossConfig"):
        t = (config.task_type or "").lower()
        if t == "binary":
            return binary_loss
        if t == "multiclass":
            return multiclass_loss
        if t == "multilabel":
            return multilabel_loss
        if t == "regression":
            return regression_loss
        raise ValueError(f"Unknown task_type: {config.task_type}")
