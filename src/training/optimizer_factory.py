"""
File Name: optimizer_factory.py
Module: TruthLens AI - Training Optimizer Factory
Description:
    Factory module for creating optimizers used in TruthLens AI training
    pipelines. Centralizes optimizer configuration to avoid hardcoded
    optimizer logic in training scripts. Supports common optimizers
    including AdamW, Adam, SGD, and Lion (if available).

Dependencies: 
    logging
    typing
    torch
Inputs:
    model_parameters: iterable of model parameters
    optimizer_name: name of optimizer
    learning_rate: learning rate value
    weight_decay: weight decay coefficient
    additional optimizer kwargs
Outputs:
    initialized torch optimizer
"""
from __future__ import annotations

import logging
from typing import Any, Iterable, List, Dict

import torch
from torch.optim import Adam, AdamW, SGD

logger = logging.getLogger(__name__)


# ---------------------------------------------------------
# OPTIONAL OPTIMIZERS
# ---------------------------------------------------------

def _try_import_lion():
    try:
        from lion_pytorch import Lion  # type: ignore
        return Lion
    except Exception:
        return None


LION_OPTIMIZER = _try_import_lion()


# ---------------------------------------------------------
# SUPPORTED OPTIMIZERS
# ---------------------------------------------------------

SUPPORTED_OPTIMIZERS = {
    "adam": Adam,
    "adamw": AdamW,
    "adamw_fused": AdamW,  # alias for fused AdamW
    "sgd": SGD,
}

if LION_OPTIMIZER:
    SUPPORTED_OPTIMIZERS["lion"] = LION_OPTIMIZER


# ---------------------------------------------------------
# PARAM GROUPING (CRITICAL)
# ---------------------------------------------------------

def _create_param_groups(
    model_parameters: Iterable,
    weight_decay: float,
) -> List[Dict[str, Any]]:
    """
    Separate parameters into decay / no-decay groups.
    """

    decay = []
    no_decay = []

    for name, param in model_parameters:
        if not param.requires_grad:
            continue

        if (
            len(param.shape) == 1
            or name.endswith(".bias")
            or "norm" in name.lower()
        ):
            no_decay.append(param)
        else:
            decay.append(param)

    return [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]


# ---------------------------------------------------------
# FACTORY
# ---------------------------------------------------------

def create_optimizer(
    model: torch.nn.Module,
    *,
    optimizer_name: str = "adamw",
    learning_rate: float = 5e-5,
    weight_decay: float = 0.01,
    betas=(0.9, 0.999),
    eps=1e-8,
    use_fused: bool = True,
    **kwargs: Any,
) -> torch.optim.Optimizer:
    """
    Create optimized optimizer with best practices.
    """

    if not isinstance(optimizer_name, str):
        raise TypeError("optimizer_name must be a string")


    optimizer_key = optimizer_name.lower()

    # Special handling for adamw_fused alias
    if optimizer_key == "adamw_fused":
        use_fused = True
        optimizer_key = "adamw"

    if optimizer_key not in SUPPORTED_OPTIMIZERS:
        raise ValueError(
            f"Unsupported optimizer '{optimizer_name}'. "
            f"Supported: {list(SUPPORTED_OPTIMIZERS.keys())}"
        )

    # ---------------------------------------------------------
    # Parameter grouping (VERY IMPORTANT)
    # ---------------------------------------------------------

    param_groups = _create_param_groups(
        model.named_parameters(),
        weight_decay=weight_decay,
    )

    optimizer_class = SUPPORTED_OPTIMIZERS[optimizer_key]

    # ---------------------------------------------------------
    # FUSED OPTIMIZER (BIG WIN)
    # ---------------------------------------------------------

    fused_available = (
        use_fused
        and torch.cuda.is_available()
        and hasattr(torch.optim, "AdamW")
        and "fused" in optimizer_class.__init__.__code__.co_varnames
    )

    logger.info(
        "Optimizer=%s | lr=%.6f | wd=%.4f | fused=%s",
        optimizer_key,
        learning_rate,
        weight_decay,
        fused_available,
    )

    # ---------------------------------------------------------
    # CREATE OPTIMIZER
    # ---------------------------------------------------------

    if optimizer_key in {"adamw", "adam"}:
        adam_common_kwargs = dict(
            lr=learning_rate,
            betas=betas,
            eps=eps,
        )

        if optimizer_key == "adamw":
            optimizer = optimizer_class(
                param_groups,
                fused=fused_available,
                **adam_common_kwargs,
                **kwargs,
            )
        else:
            optimizer = optimizer_class(
                param_groups,
                **adam_common_kwargs,
                **kwargs,
            )

    elif optimizer_key == "sgd":

        momentum = kwargs.pop("momentum", 0.9)

        optimizer = optimizer_class(
            param_groups,
            lr=learning_rate,
            momentum=momentum,
            **kwargs,
        )

    elif optimizer_key == "lion" and LION_OPTIMIZER:

        optimizer = optimizer_class(
            param_groups,
            lr=learning_rate,
            betas=betas,
            weight_decay=weight_decay,
            **kwargs,
        )

    else:
        optimizer = optimizer_class(
            param_groups,
            lr=learning_rate,
            weight_decay=weight_decay,
            **kwargs,
        )

    return optimizer


# ---------------------------------------------------------
# UTILITIES
# ---------------------------------------------------------

def list_supported_optimizers() -> list[str]:
    return sorted(SUPPORTED_OPTIMIZERS.keys())


def get_optimizer_lr(optimizer: torch.optim.Optimizer) -> float:
    if not optimizer.param_groups:
        return 0.0
    return optimizer.param_groups[0].get("lr", 0.0)