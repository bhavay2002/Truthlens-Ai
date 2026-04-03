"""
File Name: device_utils.py
Module: src.utils
Description:
    Device management utilities for TruthLens AI.

    This module provides standardized utilities for detecting available
    compute hardware (CPU, CUDA GPU, Apple MPS), moving tensors/models
    across devices, and reporting device metadata.

    Designed for reproducible ML pipelines and compatible with PyTorch
    training, inference, and distributed extensions.

Author: TruthLens Engineering
Date: 2026-04-03
Dependencies:
    - Python 3.10+
    - PyTorch

Inputs:
    - PyTorch models
    - tensors
    - nested tensor structures

Outputs:
    - torch.device objects
    - objects moved to compute device
    - device metadata
"""

from __future__ import annotations

import logging
from typing import Any

import torch


# ---------------------------------------------------------
# Logging
# ---------------------------------------------------------

logger = logging.getLogger(__name__)


# ---------------------------------------------------------
# Device Detection
# ---------------------------------------------------------


def get_device(prefer_gpu: bool = True) -> torch.device:
    """
    Detect the best available compute device.

    Priority
    --------
    1. CUDA GPU
    2. Apple MPS
    3. CPU

    Parameters
    ----------
    prefer_gpu : bool
        Whether GPU devices should be prioritized.

    Returns
    -------
    torch.device
        Selected compute device.
    """

    if prefer_gpu and torch.cuda.is_available():
        device = torch.device("cuda")
        logger.info("Using CUDA device")

    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
        logger.info("Using Apple MPS device")

    else:
        device = torch.device("cpu")
        logger.info("Using CPU device")

    return device


# ---------------------------------------------------------
# Device Metadata
# ---------------------------------------------------------


def device_name() -> str:
    """
    Return a human-readable device name.

    Returns
    -------
    str
        Device name.
    """

    if torch.cuda.is_available():
        return torch.cuda.get_device_name(0)

    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "Apple MPS"

    return "CPU"


def gpu_memory_summary() -> str:
    """
    Return a short summary of GPU memory usage.

    Returns
    -------
    str
        Memory usage summary string.
    """

    if not torch.cuda.is_available():
        return "GPU not available"

    allocated = torch.cuda.memory_allocated() / 1024**3
    reserved = torch.cuda.memory_reserved() / 1024**3
    total = torch.cuda.get_device_properties(0).total_memory / 1024**3

    return (
        f"GPU Memory — Allocated: {allocated:.2f} GB | "
        f"Reserved: {reserved:.2f} GB | "
        f"Total: {total:.2f} GB"
    )


# ---------------------------------------------------------
# Object Transfer
# ---------------------------------------------------------


def move_to_device(obj: Any, device: torch.device) -> Any:
    """
    Recursively move tensors, models, or nested structures
    to the specified device.

    Supports:
    - torch.Tensor
    - torch.nn.Module
    - dict
    - list
    - tuple

    Parameters
    ----------
    obj : Any
        Object to move.
    device : torch.device
        Target device.

    Returns
    -------
    Any
        Object placed on device.
    """

    if obj is None:
        return None

    if isinstance(obj, torch.Tensor):
        return obj.to(device)

    if isinstance(obj, torch.nn.Module):
        return obj.to(device)

    if isinstance(obj, dict):
        return {k: move_to_device(v, device) for k, v in obj.items()}

    if isinstance(obj, list):
        return [move_to_device(v, device) for v in obj]

    if isinstance(obj, tuple):
        return tuple(move_to_device(v, device) for v in obj)

    return obj


# ---------------------------------------------------------
# Distributed Training Utilities
# ---------------------------------------------------------


def get_gpu_count() -> int:
    """
    Return number of available CUDA devices.

    Returns
    -------
    int
        Number of GPUs detected.
    """

    if not torch.cuda.is_available():
        return 0

    return torch.cuda.device_count()


def set_cuda_device(device_index: int) -> None:
    """
    Set active CUDA device.

    Parameters
    ----------
    device_index : int
        GPU index.

    Raises
    ------
    RuntimeError
        If CUDA is unavailable.
    """

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA device requested but CUDA is not available")

    torch.cuda.set_device(device_index)

    logger.info("Active CUDA device set to index %d", device_index)


# ---------------------------------------------------------
# Device Summary
# ---------------------------------------------------------


def device_summary() -> dict[str, Any]:
    """
    Return structured device information.

    Returns
    -------
    dict
        Device metadata for logging/experiment tracking.
    """

    summary = {
        "device": str(get_device()),
        "device_name": device_name(),
        "gpu_count": get_gpu_count(),
        "cuda_available": torch.cuda.is_available(),
    }

    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        summary["gpu_total_memory_gb"] = round(props.total_memory / 1024**3, 2)

    return summary