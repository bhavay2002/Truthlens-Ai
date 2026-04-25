from __future__ import annotations

import logging
from typing import Dict, Any

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


# =========================================================
# BASIC COUNTS
# =========================================================

def count_parameters(model: nn.Module) -> int:
    if not isinstance(model, nn.Module):
        raise TypeError("model must be nn.Module")

    return sum(p.numel() for p in model.parameters())


def count_trainable_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def count_frozen_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if not p.requires_grad)


# =========================================================
# MEMORY ESTIMATION
# =========================================================

def estimate_model_size_mb(model: nn.Module) -> float:
    """
    Approximate model size in MB (assuming float32 unless specified).
    """

    total_params = count_parameters(model)
    bytes_per_param = 4  # float32

    return (total_params * bytes_per_param) / (1024 ** 2)


# =========================================================
# SUMMARY
# =========================================================

def parameter_summary(model: nn.Module) -> Dict[str, Any]:

    total = count_parameters(model)
    trainable = count_trainable_parameters(model)
    frozen = count_frozen_parameters(model)

    summary = {
        "total_parameters": total,
        "trainable_parameters": trainable,
        "frozen_parameters": frozen,
        "trainable_ratio": trainable / total if total > 0 else 0.0,
        "model_size_mb": estimate_model_size_mb(model),
    }

    logger.info(
        "Params | total=%d | trainable=%d | frozen=%d | size=%.2fMB",
        total,
        trainable,
        frozen,
        summary["model_size_mb"],
    )

    return summary


# =========================================================
# LAYER BREAKDOWN
# =========================================================

def layer_parameter_breakdown(model: nn.Module) -> Dict[str, Dict[str, int]]:

    breakdown: Dict[str, Dict[str, int]] = {}

    for name, module in model.named_modules():

        params = sum(p.numel() for p in module.parameters(recurse=False))
        trainable = sum(
            p.numel() for p in module.parameters(recurse=False) if p.requires_grad
        )

        if params > 0:
            breakdown[name] = {
                "total": params,
                "trainable": trainable,
                "frozen": params - trainable,
            }

    return breakdown


# =========================================================
# TOP-K HEAVIEST LAYERS
# =========================================================

def top_k_layers_by_parameters(
    model: nn.Module,
    k: int = 10,
) -> Dict[str, int]:

    layer_counts = {
        name: sum(p.numel() for p in module.parameters(recurse=False))
        for name, module in model.named_modules()
    }

    sorted_layers = sorted(
        layer_counts.items(),
        key=lambda x: x[1],
        reverse=True,
    )

    return dict(sorted_layers[:k])