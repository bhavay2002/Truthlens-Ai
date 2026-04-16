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
    return torch.autocast("cpu")


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

    if tensor.device == device:
        return tensor

    if pin_memory and tensor.device.type == "cpu":
        tensor = tensor.pin_memory()

    return tensor.to(device, non_blocking=non_blocking, dtype=dtype)


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
    inplace: bool = False,
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
            obj.to(dtype=dtype)
        return obj

    # Dict
    if isinstance(obj, dict):
        if inplace:
            for k in obj:
                obj[k] = move_to_device(
                    obj[k], device,
                    non_blocking=non_blocking,
                    pin_memory=pin_memory,
                    dtype=dtype,
                    inplace=True,
                )
            return obj

        return {
            k: move_to_device(
                v, device,
                non_blocking=non_blocking,
                pin_memory=pin_memory,
                dtype=dtype,
            )
            for k, v in obj.items()
        }

    # List
    if isinstance(obj, list):
        if inplace:
            for i in range(len(obj)):
                obj[i] = move_to_device(
                    obj[i], device,
                    non_blocking=non_blocking,
                    pin_memory=pin_memory,
                    dtype=dtype,
                    inplace=True,
                )
            return obj

        return [
            move_to_device(
                v, device,
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
                v, device,
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
        k: move_tensor(v, device,
                       non_blocking=non_blocking,
                       pin_memory=pin_memory,
                       dtype=dtype)
        if torch.is_tensor(v) else v
        for k, v in batch.items()
    }


# =========================================================
# GPU Utilities
# =========================================================

def gpu_memory_summary() -> str:

    if not torch.cuda.is_available():
        return "GPU not available"

    allocated = torch.cuda.memory_allocated() / 1024**3
    reserved = torch.cuda.memory_reserved() / 1024**3
    total = torch.cuda.get_device_properties(0).total_memory / 1024**3

    return (
        f"Allocated: {allocated:.2f}GB | "
        f"Reserved: {reserved:.2f}GB | "
        f"Total: {total:.2f}GB"
    )


def get_gpu_count() -> int:
    return torch.cuda.device_count() if torch.cuda.is_available() else 0


def set_cuda_device(index: int):
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA not available")
    torch.cuda.set_device(index)


# =========================================================
# Distributed Helpers
# =========================================================

def is_primary_process() -> bool:
    return not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0


# =========================================================
# Device Summary
# =========================================================

def device_summary() -> Dict[str, Any]:

    device = get_device()

    summary = {
        "device": str(device),
        "gpu_count": get_gpu_count(),
        "cuda": torch.cuda.is_available(),
    }

    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        summary["gpu_memory_gb"] = round(props.total_memory / 1024**3, 2)

    return summary