#src\models\training\loss_functions.py
from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F


# =========================================================
# PURE LOSS FUNCTIONS (NO TASK LOGIC)
# =========================================================

def binary_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    pos_weight: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Binary cross-entropy with optional positive-class re-weighting.

    EDGE-CASE (section 9, imbalanced binary): on heavily imbalanced data
    (e.g. ``99% / 1%`` like minority-class hate-speech detection), plain
    BCE collapses to "always predict the majority class" because the
    gradient from the rare positives is dwarfed by the negatives. The
    standard remedy is ``pos_weight`` — a per-class scalar (or tensor of
    shape ``[num_classes]``) that scales the positive term so the
    effective gradient is balanced. Exposing it here keeps callers from
    re-implementing the loss just to pass that single argument.
    """
    targets = targets.float()

    if targets.dim() == 1:
        targets = targets.unsqueeze(1)

    if logits.shape != targets.shape:
        raise RuntimeError("Shape mismatch in binary_loss")

    return F.binary_cross_entropy_with_logits(
        logits.float(),
        targets,
        pos_weight=pos_weight,
    )


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

