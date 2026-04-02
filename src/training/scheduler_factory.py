"""
File Name: scheduler_factory.py
Module: TruthLens AI - Training Scheduler Factory
Description:
    Factory module for creating learning rate schedulers used in TruthLens AI
    training pipelines. Centralizes scheduler configuration to prevent
    hardcoded learning rate logic inside training scripts.

    Supports common transformer-friendly schedulers including:
    • Linear decay with warmup
    • Cosine decay with warmup
    • Polynomial decay
    • Constant scheduler
    • Constant scheduler with warmup

Author: TruthLens Engineering Team
Date: 2026-04-02
Dependencies:
    logging
    typing
    torch
    transformers
Inputs:
    optimizer: torch optimizer instance
    scheduler_name: scheduler type
    num_training_steps: total training steps
    num_warmup_steps: warmup steps
Outputs:
    initialized learning rate scheduler
"""

from __future__ import annotations

import logging
from typing import Any

import torch

try:
    from transformers import (
        get_linear_schedule_with_warmup,
        get_cosine_schedule_with_warmup,
        get_polynomial_decay_schedule_with_warmup,
        get_constant_schedule,
        get_constant_schedule_with_warmup,
    )
except Exception as exc:  # pragma: no cover
    raise ImportError(
        "transformers is required for scheduler_factory."
    ) from exc


logger = logging.getLogger(__name__)


SUPPORTED_SCHEDULERS = {
    "linear": get_linear_schedule_with_warmup,
    "cosine": get_cosine_schedule_with_warmup,
    "polynomial": get_polynomial_decay_schedule_with_warmup,
    "constant": get_constant_schedule,
    "constant_with_warmup": get_constant_schedule_with_warmup,
}


def create_scheduler(
    optimizer: torch.optim.Optimizer,
    *,
    scheduler_name: str = "linear",
    num_training_steps: int | None = None,
    num_warmup_steps: int = 0,
    **kwargs: Any,
):
    """
    Create learning rate scheduler.

    Parameters
    ----------
    optimizer : torch.optim.Optimizer
        Optimizer instance.
    scheduler_name : str
        Scheduler type.
    num_training_steps : int
        Total number of training steps.
    num_warmup_steps : int
        Number of warmup steps.

    Returns
    -------
    Scheduler instance
    """

    if not isinstance(scheduler_name, str):
        raise TypeError("scheduler_name must be a string.")

    scheduler_key = scheduler_name.lower()

    if scheduler_key not in SUPPORTED_SCHEDULERS:
        raise ValueError(
            f"Unsupported scheduler '{scheduler_name}'. "
            f"Supported schedulers: {list(SUPPORTED_SCHEDULERS.keys())}"
        )

    scheduler_fn = SUPPORTED_SCHEDULERS[scheduler_key]

    logger.info(
        "Initializing scheduler: %s | warmup_steps=%s | training_steps=%s",
        scheduler_key,
        num_warmup_steps,
        num_training_steps,
    )

    if scheduler_key == "constant":
        return scheduler_fn(optimizer)

    if scheduler_key == "constant_with_warmup":
        return scheduler_fn(
            optimizer,
            num_warmup_steps=num_warmup_steps,
        )

    if num_training_steps is None:
        raise ValueError(
            "num_training_steps must be provided for this scheduler."
        )

    return scheduler_fn(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
        **kwargs,
    )


def list_supported_schedulers() -> list[str]:
    """
    Return list of supported scheduler names.
    """

    return sorted(SUPPORTED_SCHEDULERS.keys())