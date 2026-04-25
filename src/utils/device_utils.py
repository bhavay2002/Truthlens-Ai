from __future__ import annotations

import logging
import platform
from typing import Any, Dict
from contextlib import nullcontext

import torch

from src.utils.distributed_utils import is_primary  # ✅ centralized

logger = logging.getLogger(__name__)


# =========================================================
# DEVICE SETUP (DDP SAFE)
# =========================================================

def get_device(
    prefer_gpu: bool = True,
    local_rank: int | None = None,
) -> torch.device:
    """
    Resolve device and bind process (DDP-safe).
    """

    # -----------------------------
    # DDP MODE (CRITICAL)
    # -----------------------------
    if local_rank is not None and torch.cuda.is_available():
        torch.cuda.set_device(local_rank)  # 🔥 REQUIRED

        device = torch.device(f"cuda:{local_rank}")

        logger.info(
            "Using DDP device | local_rank=%d | device=%s",
            local_rank,
            device,
        )

        return device

    # -----------------------------
    # SINGLE GPU
    # -----------------------------
    if prefer_gpu and torch.cuda.is_available():
        device = torch.device("cuda")
        logger.info("Using CUDA device")
        return device

    # -----------------------------
    # MPS (Apple Silicon)
    # -----------------------------
    if prefer_gpu and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
        logger.info("Using MPS device")
        return device

    # -----------------------------
    # CPU
    # -----------------------------
    logger.info("Using CPU device")
    return torch.device("cpu")


def setup_device(local_rank: int | None = None) -> torch.device:
    """
    Unified device setup entrypoint.
    """
    device = get_device(local_rank=local_rank)

    if torch.cuda.is_available():
        logger.info(
            "GPU count=%d | current_device=%d",
            torch.cuda.device_count(),
            torch.cuda.current_device(),
        )

    return device


# =========================================================
# PERFORMANCE FLAGS
# =========================================================

def enable_performance_optimizations():
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True


# =========================================================
# AMP + SCALER
# =========================================================

def get_autocast_dtype() -> torch.dtype:
    if torch.cuda.is_available():
        return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    return torch.float32


def autocast_context():
    if torch.cuda.is_available():
        return torch.autocast("cuda", dtype=get_autocast_dtype())
    return nullcontext()


def get_grad_scaler(enabled: bool = True) -> torch.cuda.amp.GradScaler:
    return torch.cuda.amp.GradScaler(enabled=enabled and torch.cuda.is_available())


# =========================================================
# INFERENCE CONTEXT
# =========================================================

def inference_context():
    return torch.no_grad()


# =========================================================
# CPU BF16 SUPPORT
# =========================================================

def _cpu_supports_bf16() -> bool:
    try:
        if platform.system() == "Linux":
            with open("/proc/cpuinfo", "r") as f:
                return "bf16" in f.read().lower()
    except Exception:
        pass
    return False


# =========================================================
# TENSOR MOVE
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

    return tensor.to(device, non_blocking=use_non_blocking, dtype=dtype)


# =========================================================
# RECURSIVE MOVE (TASK SAFE)
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

    if isinstance(obj, torch.Tensor):
        return move_tensor(
            obj,
            device,
            non_blocking=non_blocking,
            pin_memory=pin_memory,
            dtype=dtype,
        )

    if isinstance(obj, dict):
        return {
            k: move_to_device(v, device, non_blocking=non_blocking, pin_memory=pin_memory, dtype=dtype)
            if isinstance(v, (torch.Tensor, dict, list, tuple))
            else v  # do not move strings like "task"
            for k, v in obj.items()
        }

    if isinstance(obj, list):
        return [
            move_to_device(v, device, non_blocking=non_blocking, pin_memory=pin_memory, dtype=dtype)
            for v in obj
        ]

    if isinstance(obj, tuple):
        return tuple(
            move_to_device(v, device, non_blocking=non_blocking, pin_memory=pin_memory, dtype=dtype)
            for v in obj
        )

    if isinstance(obj, torch.nn.Module):
        return obj.to(device=device)

    return obj


# =========================================================
# BATCH MOVE
# =========================================================

def move_batch(
    batch: Dict[str, Any],
    device: torch.device,
    *,
    pin_memory: bool = False,
    non_blocking: bool = True,
) -> Dict[str, Any]:

    if not isinstance(batch, dict):
        raise TypeError("Batch must be dict")

    return move_to_device(
        batch,
        device,
        non_blocking=non_blocking,
        pin_memory=pin_memory,
    )


# =========================================================
# GPU UTILITIES
# =========================================================

def gpu_memory_summary(device_index: int = 0) -> str:

    if not torch.cuda.is_available():
        return "GPU not available"

    allocated = torch.cuda.memory_allocated(device_index) / 1024**3
    total = torch.cuda.get_device_properties(device_index).total_memory / 1024**3

    return f"GPU {device_index} | {allocated:.2f}/{total:.2f} GB"


def get_gpu_count() -> int:
    return torch.cuda.device_count() if torch.cuda.is_available() else 0


# =========================================================
# PROCESS ROLE (DELEGATED)
# =========================================================

def is_primary_process() -> bool:
    """
    Backward-compatible alias.
    Prefer: src.utils.distributed_utils.is_primary
    """
    return is_primary()