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
    """
    Multi-label binary cross-entropy with element-wise ignore semantics.

    LOSS-4: Ignore-mask contract (now explicit).

    Multi-label targets are conventionally float in {0.0, 1.0}, so the
    integer sentinel used by ``multiclass_loss`` (``-100``) does not
    naturally appear in normal data — meaning the original mask
    ``targets != -100.0`` was effectively always all-True and the
    ``ignore_index`` argument was a no-op for the common case.

    The supported sentinels are now documented and consistently masked:
      * ``ignore_index`` (default ``-100.0``): explicit float sentinel.
        Pass any value (e.g. ``float('nan')``) to override.
      * ``NaN`` targets: ALWAYS treated as ignored. NaN labels would
        otherwise propagate through ``BCEWithLogits`` and corrupt the
        loss silently with no traceback.

    Both rules combine — an element is included iff it is finite AND not
    equal to ``ignore_index``.
    """

    targets = targets.float()

    if logits.shape != targets.shape:
        raise RuntimeError("Shape mismatch in multilabel_loss")

    # Finite-and-not-sentinel mask. ``isfinite`` masks NaN AND ±inf.
    mask = torch.isfinite(targets) & (targets != ignore_index)

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
