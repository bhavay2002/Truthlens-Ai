from __future__ import annotations

import logging
from typing import Dict, Iterable, Optional, Tuple

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
    check_dtypes: bool = False,
    expected_shapes: Optional[Dict[str, Tuple[int, ...]]] = None,
) -> None:
    """
    Validate checkpoint integrity.

    Parameters
    ----------
    state_dict : dict
        Model weights

    required_prefixes : list[str]
        Expected module prefixes

    strict : bool
        Raise error on missing components

    check_shapes : bool
        Validate tensor shapes

    check_dtypes : bool
        Validate dtype consistency

    expected_shapes : dict[str, tuple]
        Optional exact shape expectations
    """

    # -----------------------------------------------------
    # BASIC VALIDATION
    # -----------------------------------------------------

    if not isinstance(state_dict, dict) or not state_dict:
        raise ValueError("Invalid or empty state_dict")

    prefixes = list(required_prefixes or REQUIRED_PREFIXES)

    # -----------------------------------------------------
    # FINITE CHECK
    # -----------------------------------------------------

    for name, tensor in state_dict.items():

        if not torch.is_tensor(tensor):
            continue

        if tensor.is_floating_point():
            if not torch.isfinite(tensor).all():
                raise ValueError(f"Non-finite values detected in: {name}")

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
    # SHAPE CHECK
    # -----------------------------------------------------

    if check_shapes:

        for name, tensor in state_dict.items():

            if not torch.is_tensor(tensor):
                continue

            if tensor.numel() == 0:
                raise ValueError(f"Empty tensor detected: {name}")

            if any(dim <= 0 for dim in tensor.shape):
                raise ValueError(f"Invalid shape in {name}: {tensor.shape}")

    # -----------------------------------------------------
    # EXACT SHAPE MATCHING (ADVANCED)
    # -----------------------------------------------------

    if expected_shapes:

        for name, expected_shape in expected_shapes.items():

            if name not in state_dict:
                continue

            actual = tuple(state_dict[name].shape)

            if actual != expected_shape:
                raise ValueError(
                    f"Shape mismatch for {name}: "
                    f"expected={expected_shape}, got={actual}"
                )

    # -----------------------------------------------------
    # DTYPE CHECK (OPTIONAL)
    # -----------------------------------------------------

    if check_dtypes:

        dtypes = {
            tensor.dtype
            for tensor in state_dict.values()
            if torch.is_tensor(tensor)
        }

        if len(dtypes) > 1:
            logger.warning(f"Mixed dtypes detected: {dtypes}")

    # -----------------------------------------------------
    # DEVICE CHECK (DEBUGGING)
    # -----------------------------------------------------

    devices = {
        str(tensor.device)
        for tensor in state_dict.values()
        if torch.is_tensor(tensor)
    }

    if len(devices) > 1:
        logger.warning(f"Mixed devices in checkpoint: {devices}")

    logger.info(
        "Checkpoint validation passed | tensors=%d | prefixes=%d",
        len(state_dict),
        len(prefixes),
    )