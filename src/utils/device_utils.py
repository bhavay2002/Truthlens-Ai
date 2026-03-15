"""
File: device_utils.py

Purpose
-------
Device utilities for TruthLens AI.

This module handles detection and usage of CPU or GPU
devices for training and inference.
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
# Detect Device
# ---------------------------------------------------------

def get_device() -> torch.device:
    """
    Detect available compute device.

    Returns
    -------
    torch.device
        CUDA device if available, otherwise CPU.
    """

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    logger.info("Using device: %s", device)

    return device


# ---------------------------------------------------------
# Move Object to Device
# ---------------------------------------------------------

def move_to_device(obj: Any, device: torch.device):
    """
    Move PyTorch model or tensor to device.
    """

    if hasattr(obj, "to"):

        return obj.to(device)

    return obj


# ---------------------------------------------------------
# Device Name
# ---------------------------------------------------------

def device_name() -> str:
    """
    Return human-readable device name.
    """

    if torch.cuda.is_available():

        return torch.cuda.get_device_name(0)

    return "CPU"