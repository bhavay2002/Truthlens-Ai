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
from typing import Any, Iterable

import torch
from torch.optim import Adam, AdamW, SGD

logger = logging.getLogger(__name__)


SUPPORTED_OPTIMIZERS = {
    "adam": Adam,
    "adamw": AdamW,
    "sgd": SGD,
}


def _try_import_lion():
    """
    Attempt to import Lion optimizer if available.
    """

    try:
        from lion_pytorch import Lion  # type: ignore
        return Lion
    except Exception:
        return None


LION_OPTIMIZER = _try_import_lion()

if LION_OPTIMIZER is not None:
    SUPPORTED_OPTIMIZERS["lion"] = LION_OPTIMIZER


def create_optimizer(
    model_parameters: Iterable,
    *,
    optimizer_name: str = "adamw",
    learning_rate: float = 5e-5,
    weight_decay: float = 0.0,
    **kwargs: Any,
) -> torch.optim.Optimizer:
    """
    Create and return a PyTorch optimizer.

    Parameters
    ----------
    model_parameters : Iterable
        Model parameters from model.parameters()
    optimizer_name : str
        Name of optimizer (adamw, adam, sgd, lion)
    learning_rate : float
        Learning rate
    weight_decay : float
        Weight decay coefficient
    kwargs : Any
        Additional optimizer parameters

    Returns
    -------
    torch.optim.Optimizer
    """

    if not isinstance(optimizer_name, str):
        raise TypeError("optimizer_name must be a string")

    optimizer_key = optimizer_name.lower()

    if optimizer_key not in SUPPORTED_OPTIMIZERS:
        raise ValueError(
            f"Unsupported optimizer '{optimizer_name}'. "
            f"Supported optimizers: {list(SUPPORTED_OPTIMIZERS.keys())}"
        )

    optimizer_class = SUPPORTED_OPTIMIZERS[optimizer_key]

    logger.info(
        "Initializing optimizer: %s | lr=%.6f | weight_decay=%.6f",
        optimizer_key,
        learning_rate,
        weight_decay,
    )

    optimizer = optimizer_class(
        model_parameters,
        lr=learning_rate,
        weight_decay=weight_decay,
        **kwargs,
    )

    return optimizer


def list_supported_optimizers() -> list[str]:
    """
    Return list of supported optimizer names.
    """

    return sorted(SUPPORTED_OPTIMIZERS.keys())