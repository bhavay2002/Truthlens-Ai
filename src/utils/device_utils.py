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
from typing import Any, Dict
from contextlib import nullcontext

import torch

logger = logging.getLogger(__name__)


# =========================================================
# Device Detection
# =========================================================

def get_device(prefer_gpu: bool = True) -> torch.device:

    if prefer_gpu and torch.cuda.is_available():
        return torch.device("cuda")

    if prefer_gpu and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")

    return torch.device("cpu")


# =========================================================
# AMP Utilities
# =========================================================

def get_autocast_dtype() -> torch.dtype:
    if torch.cuda.is_available():
        # bf16 preferred on modern GPUs
        return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    return torch.float32


def autocast_context():
    if torch.cuda.is_available():
        return torch.autocast("cuda", dtype=get_autocast_dtype())
    if hasattr(torch, "autocast"):
        try:
            return torch.autocast("cpu", dtype=torch.bfloat16)
        except Exception:
            return nullcontext()
    return nullcontext()


# =========================================================
# Fast Tensor Movement
# =========================================================

def move_tensor(
    tensor: torch.Tensor,
    device: torch.device,
    *,
    non_blocking: bool = True,
    pin_memory: bool = False,
    dtype: torch.dtype | None = None,
) -> torch.Tensor:

    if not isinstance(tensor, torch.Tensor):
        return tensor

    if tensor.device == device and dtype is None:
        return tensor

    if pin_memory and tensor.device.type == "cpu" and device.type == "cuda":
        tensor = tensor.pin_memory()

    use_non_blocking = (
        non_blocking and tensor.device.type == "cpu" and device.type == "cuda"
    )

    if dtype is not None and torch.is_floating_point(tensor):
        return tensor.to(device, non_blocking=use_non_blocking, dtype=dtype)

    return tensor.to(device, non_blocking=use_non_blocking)


# =========================================================
# Recursive Movement (Optimized)
# =========================================================

def move_to_device(
    obj: Any,
    device: torch.device,
    *,
    non_blocking: bool = True,
    pin_memory: bool = False,
    dtype: torch.dtype | None = None,
) -> Any:

    if obj is None:
        return None

    # Tensor
    if isinstance(obj, torch.Tensor):
        return move_tensor(
            obj,
            device,
            non_blocking=non_blocking,
            pin_memory=pin_memory,
            dtype=dtype,
        )

    # Model
    if isinstance(obj, torch.nn.Module):
        obj.to(device)
        if dtype is not None:
            for p in obj.parameters():
                if torch.is_floating_point(p):
                    p.data = p.data.to(dtype=dtype)
        return obj

    # Dict
    if isinstance(obj, dict):
        return {
            k: move_to_device(
                v,
                device,
                non_blocking=non_blocking,
                pin_memory=pin_memory,
                dtype=dtype,
            )
            for k, v in obj.items()
        }

    # List
    if isinstance(obj, list):
        return [
            move_to_device(
                v,
                device,
                non_blocking=non_blocking,
                pin_memory=pin_memory,
                dtype=dtype,
            )
            for v in obj
        ]

    # Tuple
    if isinstance(obj, tuple):
        return tuple(
            move_to_device(
                v,
                device,
                non_blocking=non_blocking,
                pin_memory=pin_memory,
                dtype=dtype,
            )
            for v in obj
        )

    return obj


# =========================================================
# Batch Optimization
# =========================================================

def move_batch(
    batch: Dict[str, Any],
    device: torch.device,
    *,
    pin_memory: bool = True,
    non_blocking: bool = True,
    dtype: torch.dtype | None = None,
) -> Dict[str, Any]:

    return {
        k: move_to_device(
            v,
            device,
            non_blocking=non_blocking,
            pin_memory=pin_memory,
            dtype=dtype,
        )
        for k, v in batch.items()
    }


# =========================================================
# GPU Utilities
# =========================================================

def gpu_memory_summary(device_index: int = 0) -> str:

    if not torch.cuda.is_available():
        return "GPU not available"

    count = torch.cuda.device_count()
    if device_index >= count:
        raise ValueError(f"Invalid GPU index {device_index}")

    allocated = torch.cuda.memory_allocated(device_index) / 1024**3
    reserved = torch.cuda.memory_reserved(device_index) / 1024**3
    total = torch.cuda.get_device_properties(device_index).total_memory / 1024**3

    return (
        f"GPU {device_index} | "
        f"Allocated: {allocated:.2f}GB | "
        f"Reserved: {reserved:.2f}GB | "
        f"Total: {total:.2f}GB"
    )


def get_gpu_count() -> int:
    return torch.cuda.device_count() if torch.cuda.is_available() else 0


def set_cuda_device(index: int):
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA not available")
    count = torch.cuda.device_count()
    if index < 0 or index >= count:
        raise ValueError(
            f"CUDA device index out of range: {index}, available: 0..{count - 1}"
        )
    torch.cuda.set_device(index)


# =========================================================
# Distributed Helpers
# =========================================================

def is_primary_process() -> bool:
    if not torch.distributed.is_available():
        return True

    if not torch.distributed.is_initialized():
        return True

    return torch.distributed.get_rank() == 0


# =========================================================
# Device Summary
# =========================================================

def device_name(device: torch.device | None = None) -> str:
    """Return a human-readable name for the current compute device."""
    device = device or get_device()

    if device.type == "cuda":
        try:
            return torch.cuda.get_device_name(device.index or 0)
        except Exception:
            return "CUDA GPU"

    if device.type == "mps":
        return "Apple MPS"

    return "CPU"


def device_summary(device: torch.device | None = None) -> Dict[str, Any]:
    device = device or get_device()

    summary = {
        "device": str(device),
        "device_name": device_name(device),
        "gpu_count": get_gpu_count(),
        "cuda": torch.cuda.is_available(),
    }

    if device.type == "cuda":
        idx = device.index or 0
        props = torch.cuda.get_device_properties(idx)
        summary["gpu_memory_gb"] = round(props.total_memory / 1024**3, 2)

    return summary