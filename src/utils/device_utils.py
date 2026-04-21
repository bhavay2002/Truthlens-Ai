from __future__ import annotations

import logging
import platform
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
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return nullcontext()
    if hasattr(torch, "autocast") and _cpu_supports_bf16():
        try:
            return torch.autocast("cpu", dtype=torch.bfloat16)
        except Exception:
            return nullcontext()
    return nullcontext()


def _cpu_supports_bf16() -> bool:
    if not hasattr(torch.backends, "cpu"):
        return False

    if hasattr(torch.backends.cpu, "has_bf16"):
        if torch.backends.cpu.has_bf16:
            return True

    try:
        if platform.system() == "Linux":
            with open("/proc/cpuinfo", "r", encoding="utf-8") as cpuinfo_file:
                cpuinfo = cpuinfo_file.read().lower()
            return "avx512_bf16" in cpuinfo or "amx_bf16" in cpuinfo
    except Exception:
        return False

    return False


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
        non_blocking
        and tensor.device.type == "cpu"
        and device.type == "cuda"
        and tensor.is_pinned()
    )

    if dtype is not None:
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
        return obj.to(device=device, dtype=dtype)

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
    pin_memory: bool = False,
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
    return (
        not torch.distributed.is_available()
        or not torch.distributed.is_initialized()
        or torch.distributed.get_rank() == 0
    )


# =========================================================
# Device Summary
# =========================================================

def device_name(device: torch.device | None = None) -> str:
    """Return a human-readable name for the current compute device."""
    device = device or get_device()

    if device.type == "cuda":
        try:
            idx = device.index if device.index is not None else 0
            return torch.cuda.get_device_name(idx)
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
        idx = device.index if device.index is not None else 0
        props = torch.cuda.get_device_properties(idx)
        summary["gpu_memory_gb"] = round(props.total_memory / 1024**3, 2)

    return summary