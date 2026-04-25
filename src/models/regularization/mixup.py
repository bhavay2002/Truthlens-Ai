from __future__ import annotations

from typing import Tuple, Optional

import numpy as np
import torch
import torch.nn.functional as F


# =========================================================
# MIXUP UTILS
# =========================================================

def _sample_lambda(alpha: float, size: int = 1) -> float:
    if alpha <= 0:
        return 1.0
    lam = np.random.beta(alpha, alpha, size=size)
    return float(lam[0] if size == 1 else lam)


def _shuffle_indices(batch_size: int, device: torch.device) -> torch.Tensor:
    return torch.randperm(batch_size, device=device)


# =========================================================
# STANDARD MIXUP (INPUT-LEVEL)
# =========================================================

def mixup(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    alpha: float = 0.2,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
    """
    Perform MixUp on inputs and targets.

    Args:
        inputs: (B, ...)
        targets: (B, C) or (B,)
        alpha: Beta distribution parameter

    Returns:
        mixed_inputs
        targets_a
        targets_b
        lam
    """

    device = inputs.device
    batch_size = inputs.size(0)

    lam = _sample_lambda(alpha)

    index = _shuffle_indices(batch_size, device)

    mixed_inputs = lam * inputs + (1 - lam) * inputs[index]

    targets_a = targets
    targets_b = targets[index]

    return mixed_inputs, targets_a, targets_b, lam


# =========================================================
# EMBEDDING-LEVEL MIXUP
# =========================================================

def embedding_mixup(
    embeddings: torch.Tensor,
    targets: torch.Tensor,
    alpha: float = 0.2,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
    """
    MixUp applied to hidden representations.

    Args:
        embeddings: (B, H) or (B, T, H)
        targets: labels

    Returns:
        mixed_embeddings, targets_a, targets_b, lam
    """

    device = embeddings.device
    batch_size = embeddings.size(0)

    lam = _sample_lambda(alpha)

    index = _shuffle_indices(batch_size, device)

    mixed_embeddings = lam * embeddings + (1 - lam) * embeddings[index]

    return mixed_embeddings, targets, targets[index], lam


# =========================================================
# LOSS WRAPPER
# =========================================================

def mixup_loss(
    criterion,
    preds: torch.Tensor,
    targets_a: torch.Tensor,
    targets_b: torch.Tensor,
    lam: float,
) -> torch.Tensor:
    """
    Compute MixUp loss.

    Args:
        criterion: loss function
        preds: model predictions
        targets_a: original targets
        targets_b: shuffled targets
        lam: mix coefficient
    """

    loss_a = criterion(preds, targets_a)
    loss_b = criterion(preds, targets_b)

    return lam * loss_a + (1 - lam) * loss_b


# =========================================================
# MULTILABEL SUPPORT
# =========================================================

def mixup_multilabel(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    alpha: float = 0.2,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    MixUp for multilabel classification.

    Targets are mixed directly.

    Returns:
        mixed_inputs, mixed_targets
    """

    device = inputs.device
    batch_size = inputs.size(0)

    lam = _sample_lambda(alpha)

    index = _shuffle_indices(batch_size, device)

    mixed_inputs = lam * inputs + (1 - lam) * inputs[index]
    mixed_targets = lam * targets + (1 - lam) * targets[index]

    return mixed_inputs, mixed_targets


# =========================================================
# TOKEN-LEVEL MIXUP (NLP)
# =========================================================

def token_mixup(
    embeddings: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    alpha: float = 0.2,
) -> Tuple[torch.Tensor, float]:
    """
    MixUp at token level (sequence-wise interpolation).

    Args:
        embeddings: (B, T, H)
        attention_mask: optional mask

    Returns:
        mixed_embeddings, lam
    """

    device = embeddings.device
    batch_size = embeddings.size(0)

    lam = _sample_lambda(alpha)
    index = _shuffle_indices(batch_size, device)

    mixed = lam * embeddings + (1 - lam) * embeddings[index]

    if attention_mask is not None:
        mask = attention_mask.unsqueeze(-1)
        mixed = mixed * mask

    return mixed, lam