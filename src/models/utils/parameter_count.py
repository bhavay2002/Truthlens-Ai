"""
File Name: parameter_count.py
Module: models.utils
Description:
    Provides utilities for inspecting the number of parameters in PyTorch
    models used within the TruthLens AI system. These utilities help
    developers understand model size, memory footprint, and trainable
    parameter distribution.

    Functions include:
        • total parameter count
        • trainable parameter count
        • frozen parameter count
        • per-layer parameter summaries

Dependencies:
    logging
    typing
    torch
    torch.nn
Inputs:
    PyTorch model
Outputs:
    Parameter statistics and summaries
"""

from __future__ import annotations

import logging
from typing import Dict

import torch.nn as nn

logger = logging.getLogger(__name__)


def count_parameters(model: nn.Module) -> int:
    """
    Count total parameters in a model.

    Parameters
    ----------
    model : nn.Module
        PyTorch model.

    Returns
    -------
    int
        Total number of parameters.
    """

    if not isinstance(model, nn.Module):
        raise TypeError("model must be an instance of torch.nn.Module")

    total = sum(p.numel() for p in model.parameters())

    logger.debug("Total parameters: %d", total)

    return total


def count_trainable_parameters(model: nn.Module) -> int:
    """
    Count trainable parameters.

    Parameters
    ----------
    model : nn.Module

    Returns
    -------
    int
    """

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

    logger.debug("Trainable parameters: %d", trainable)

    return trainable


def count_frozen_parameters(model: nn.Module) -> int:
    """
    Count non-trainable parameters.

    Parameters
    ----------
    model : nn.Module

    Returns
    -------
    int
    """

    frozen = sum(p.numel() for p in model.parameters() if not p.requires_grad)

    logger.debug("Frozen parameters: %d", frozen)

    return frozen


def parameter_summary(model: nn.Module) -> Dict[str, int]:
    """
    Generate a summary of model parameters.

    Parameters
    ----------
    model : nn.Module

    Returns
    -------
    Dict[str, int]
        Dictionary with parameter statistics.
    """

    total = count_parameters(model)
    trainable = count_trainable_parameters(model)
    frozen = count_frozen_parameters(model)

    summary = {
        "total_parameters": total,
        "trainable_parameters": trainable,
        "frozen_parameters": frozen,
    }

    logger.info(
        "Model parameters | total=%d | trainable=%d | frozen=%d",
        total,
        trainable,
        frozen,
    )

    return summary


def layer_parameter_breakdown(model: nn.Module) -> Dict[str, int]:
    """
    Return parameter count per layer/module.

    Parameters
    ----------
    model : nn.Module

    Returns
    -------
    Dict[str, int]
        Mapping of module names to parameter counts.
    """

    breakdown: Dict[str, int] = {}

    for name, module in model.named_modules():
        params = sum(p.numel() for p in module.parameters(recurse=False))

        if params > 0:
            breakdown[name] = params

    return breakdown