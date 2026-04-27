from __future__ import annotations

import logging
from typing import Iterable, Optional, Dict, Any, Literal

import torch
from torch.optim import Optimizer
from torch.optim import Adam, AdamW, SGD, RMSprop, Adagrad

logger = logging.getLogger(__name__)


OptimizerType = Literal[
    "adam",
    "adamw",
    "sgd",
    "rmsprop",
    "adagrad",
]


# =========================================================
# PARAM GROUPING (WEIGHT DECAY SPLIT)
# =========================================================

def build_parameter_groups(
    model: torch.nn.Module,
    weight_decay: float = 0.01,
    no_decay_keywords: tuple[str, ...] = (
        "bias",
        "LayerNorm.weight",
        "layer_norm.weight",
        "norm.weight",
    ),
) -> list[dict[str, Any]]:
    """
    Splits parameters into decay / no-decay groups.

    Returns:
        [
            {"params": [...], "weight_decay": wd},
            {"params": [...], "weight_decay": 0.0},
        ]
    """

    decay_params = []
    no_decay_params = []

    for name, param in model.named_parameters():

        if not param.requires_grad:
            continue

        if any(nd in name for nd in no_decay_keywords):
            no_decay_params.append(param)
        else:
            decay_params.append(param)

    return [
        {"params": decay_params, "weight_decay": weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0},
    ]


# =========================================================
# OPTIMIZER FACTORY
# =========================================================

def create_optimizer(
    model: torch.nn.Module,
    optimizer_type: OptimizerType = "adamw",
    learning_rate: float = 2e-5,
    weight_decay: float = 0.01,
    betas: tuple[float, float] = (0.9, 0.999),
    eps: float = 1e-8,
    momentum: float = 0.9,
    use_param_groups: bool = True,
    custom_params: Optional[Iterable[torch.nn.Parameter]] = None,
) -> Optimizer:
    """
    Create optimizer with best practices.

    Supports:
        - weight decay exclusion
        - custom parameter groups
        - multiple optimizers
    """

    if custom_params is not None:
        params = list(custom_params)

    elif use_param_groups:
        params = build_parameter_groups(
            model=model,
            weight_decay=weight_decay,
        )

    else:
        params = model.parameters()

    optimizer_type = optimizer_type.lower()

    logger.info(f"[OPTIMIZER] Using {optimizer_type}")

    if optimizer_type == "adamw":
        return AdamW(
            params,
            lr=learning_rate,
            betas=betas,
            eps=eps,
        )

    elif optimizer_type == "adam":
        return Adam(
            params,
            lr=learning_rate,
            betas=betas,
            eps=eps,
        )

    elif optimizer_type == "sgd":
        return SGD(
            params,
            lr=learning_rate,
            momentum=momentum,
            weight_decay=weight_decay,
        )

    elif optimizer_type == "rmsprop":
        return RMSprop(
            params,
            lr=learning_rate,
            momentum=momentum,
        )

    elif optimizer_type == "adagrad":
        return Adagrad(
            params,
            lr=learning_rate,
        )

    else:
        raise ValueError(f"Unsupported optimizer type: {optimizer_type}")


# =========================================================
# MULTI-OPTIMIZER (ADVANCED)
# =========================================================

class MultiOptimizer:
    """
    Supports multiple optimizers for different parts of the model.
    """

    def __init__(self, optimizers: Dict[str, Optimizer]) -> None:
        self.optimizers = optimizers

    def zero_grad(self) -> None:
        for opt in self.optimizers.values():
            opt.zero_grad()

    def step(self) -> None:
        for opt in self.optimizers.values():
            opt.step()

    def state_dict(self) -> Dict[str, Any]:
        return {k: v.state_dict() for k, v in self.optimizers.items()}

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        for k, v in state_dict.items():
            if k in self.optimizers:
                self.optimizers[k].load_state_dict(v)

    def get_lrs(self) -> Dict[str, float]:
        return {
            name: opt.param_groups[0]["lr"]
            for name, opt in self.optimizers.items()
        }


def build_optimizer(
    model: torch.nn.Module,
    lr: float = 2e-5,
    weight_decay: float = 0.01,
    optimizer_type: OptimizerType = "adamw",
    **kwargs: Any,
) -> Optimizer:

    return create_optimizer(
        model=model,
        optimizer_type=optimizer_type,
        learning_rate=lr,
        weight_decay=weight_decay,
        **kwargs,
    )