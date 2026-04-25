from __future__ import annotations

import logging
from typing import Dict, Iterable, Optional

import torch

logger = logging.getLogger(__name__)


# =========================================================
# CONFIG
# =========================================================

REQUIRED_PREFIXES = [
    "encoder",
    "bias_head",
    "ideology_head",
    "propaganda_head",
    "narrative_head",
    "emotion_head",
]


# =========================================================
# CORE VALIDATION
# =========================================================

def validate_checkpoint(
    state_dict: Dict[str, torch.Tensor],
    *,
    required_prefixes: Optional[Iterable[str]] = None,
    strict: bool = True,
    check_shapes: bool = False,
) -> None:

    if not isinstance(state_dict, dict) or not state_dict:
        raise ValueError("Invalid or empty state_dict")

    prefixes = list(required_prefixes or REQUIRED_PREFIXES)

    # -----------------------------------------------------
    # FINITE CHECK
    # -----------------------------------------------------

    for k, v in state_dict.items():

        if not torch.is_tensor(v):
            continue

        if v.is_floating_point():
            if not torch.isfinite(v).all():
                raise ValueError(f"Non-finite values in: {k}")

    # -----------------------------------------------------
    # STRUCTURE CHECK
    # -----------------------------------------------------

    missing = [
        p for p in prefixes
        if not any(k.startswith(p) for k in state_dict.keys())
    ]

    if missing:

        msg = f"Missing required components: {missing}"

        if strict:
            raise ValueError(msg)

        logger.warning(msg)

    # -----------------------------------------------------
    # SHAPE CHECK (OPTIONAL)
    # -----------------------------------------------------

    if check_shapes:

        for k, v in state_dict.items():

            if not torch.is_tensor(v):
                continue

            if v.numel() == 0:
                raise ValueError(f"Empty tensor: {k}")

            if any(dim <= 0 for dim in v.shape):
                raise ValueError(f"Invalid shape in {k}: {v.shape}")

    logger.info("Checkpoint validation passed")