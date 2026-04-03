"""
File Name: weight_initialization.py
Module: models.initialization
Description:
    Provides weight initialization utilities for neural network models used
    in the TruthLens AI system. The module implements commonly used
    initialization strategies for deep learning models including Xavier,
    Kaiming, normal, and uniform initialization.

    These utilities ensure consistent model initialization across training
    pipelines and support deterministic experiments when combined with
    seed control.

Dependencies:
    logging
    typing
    torch
    torch.nn
Inputs:
    PyTorch model modules
Outputs:
    Initialized model parameters
"""

from __future__ import annotations

import logging
from typing import Optional

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


def initialize_weights(
    model: nn.Module,
    method: str = "xavier",
    bias_value: float = 0.0,
) -> None:
    """
    Initialize model weights.

    Parameters
    ----------
    model : nn.Module
        PyTorch model whose parameters will be initialized.
    method : str
        Initialization method ('xavier', 'kaiming', 'normal', 'uniform').
    bias_value : float
        Constant value used to initialize bias parameters.
    """

    if not isinstance(model, nn.Module):
        raise TypeError("model must be an instance of torch.nn.Module")

    logger.info("Initializing model weights using '%s' method", method)

    for module in model.modules():

        if isinstance(module, (nn.Linear, nn.Conv1d, nn.Conv2d, nn.Conv3d)):

            if method == "xavier":
                nn.init.xavier_uniform_(module.weight)

            elif method == "kaiming":
                nn.init.kaiming_uniform_(module.weight, nonlinearity="relu")

            elif method == "normal":
                nn.init.normal_(module.weight, mean=0.0, std=0.02)

            elif method == "uniform":
                nn.init.uniform_(module.weight, a=-0.1, b=0.1)

            else:
                raise ValueError(f"Unsupported initialization method: {method}")

            if module.bias is not None:
                nn.init.constant_(module.bias, bias_value)

        elif isinstance(module, nn.Embedding):

            if method in {"xavier", "kaiming"}:
                nn.init.normal_(module.weight, mean=0.0, std=0.02)

            elif method == "normal":
                nn.init.normal_(module.weight, mean=0.0, std=0.02)

            elif method == "uniform":
                nn.init.uniform_(module.weight, a=-0.1, b=0.1)

        elif isinstance(module, nn.LayerNorm):

            nn.init.constant_(module.bias, 0.0)
            nn.init.constant_(module.weight, 1.0)


def reset_module_parameters(module: nn.Module) -> None:
    """
    Reset parameters of a module using its internal reset method if available.

    Parameters
    ----------
    module : nn.Module
        Module to reset.
    """

    if hasattr(module, "reset_parameters"):
        module.reset_parameters()


def apply_weight_initialization(
    model: nn.Module,
    method: str = "xavier",
) -> None:
    """
    Apply weight initialization to the entire model.

    Parameters
    ----------
    model : nn.Module
        Model to initialize.
    method : str
        Initialization method.
    """

    initialize_weights(model, method=method)

    logger.info("Weight initialization applied to model")