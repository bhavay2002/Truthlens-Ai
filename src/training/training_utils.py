#src\models\training\training_utils.py
from __future__ import annotations

import logging
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Dict, Iterable, Optional, Any

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


# =========================================================
# PRECISION CONTROL
# =========================================================

def configure_training_precision(
    *,
    allow_tf32: bool = True,
    matmul_precision: str = "high",
) -> None:

    if torch.cuda.is_available():
        try:
            torch.backends.cuda.matmul.allow_tf32 = bool(allow_tf32)
            torch.backends.cudnn.allow_tf32 = bool(allow_tf32)
        except Exception as exc:
            logger.debug("TF32 config skipped: %s", exc)

    try:
        torch.set_float32_matmul_precision(matmul_precision)
    except Exception as exc:
        logger.debug("Matmul precision skipped: %s", exc)


@contextmanager
def training_precision(
    *,
    allow_tf32: bool = True,
    matmul_precision: str = "high",
):
    prev_tf32_matmul = None
    prev_tf32_cudnn = None

    if torch.cuda.is_available():
        prev_tf32_matmul = torch.backends.cuda.matmul.allow_tf32
        prev_tf32_cudnn = torch.backends.cudnn.allow_tf32

    configure_training_precision(
        allow_tf32=allow_tf32,
        matmul_precision=matmul_precision,
    )

    try:
        yield
    finally:
        if torch.cuda.is_available() and prev_tf32_matmul is not None:
            torch.backends.cuda.matmul.allow_tf32 = prev_tf32_matmul
            torch.backends.cudnn.allow_tf32 = prev_tf32_cudnn


# =========================================================
# DEVICE
# =========================================================

def get_device(device: Optional[str] = None) -> torch.device:

    if device:
        return torch.device(device)

    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def move_batch_to_device(
    batch: Any,
    device: torch.device,
    non_blocking: bool = True,
) -> Any:

    if isinstance(batch, torch.Tensor):
        use_nb = (
            non_blocking
            and batch.device.type == "cpu"
            and device.type == "cuda"
            and batch.is_pinned()
        )
        return batch.to(device, non_blocking=use_nb)

    if isinstance(batch, dict):
        return {
            k: move_batch_to_device(v, device, non_blocking)
            for k, v in batch.items()
        }

    if isinstance(batch, (list, tuple)):
        return type(batch)(
            move_batch_to_device(v, device, non_blocking)
            for v in batch
        )

    return batch


# =========================================================
# GRADIENT UTILITIES
# =========================================================

def clip_gradients(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    max_norm: Optional[float],
    scaler: Optional[torch.cuda.amp.GradScaler] = None,
) -> float:

    if max_norm is None:
        return 0.0

    if max_norm <= 0:
        raise ValueError("max_norm must be > 0")

    if scaler is not None:
        scaler.unscale_(optimizer)  # ✅ FIXED

    total_norm = torch.nn.utils.clip_grad_norm_(
        model.parameters(),
        max_norm,
    )

    return float(total_norm)


def zero_gradients(optimizer: torch.optim.Optimizer) -> None:
    optimizer.zero_grad(set_to_none=True)


# =========================================================
# BATCH SIZE
# =========================================================

def compute_batch_size(batch: Any) -> int:

    if isinstance(batch, dict) and "input_ids" in batch:
        return batch["input_ids"].shape[0]

    if isinstance(batch, torch.Tensor):
        return batch.shape[0] if batch.ndim > 0 else 1

    if isinstance(batch, dict):
        for v in batch.values():
            if isinstance(v, torch.Tensor) and v.ndim > 0:
                return v.shape[0]

    if isinstance(batch, (list, tuple)):
        for v in batch:
            if isinstance(v, torch.Tensor) and v.ndim > 0:
                return v.shape[0]

    # ✅ safer fallback
    return 1


# =========================================================
# TENSOR UTILITIES
# =========================================================

def detach_tensor_dict(data: Any, to_cpu: bool = True) -> Any:

    if isinstance(data, torch.Tensor):
        t = data.detach()
        return t.cpu() if to_cpu else t

    if isinstance(data, dict):
        return {k: detach_tensor_dict(v, to_cpu) for k, v in data.items()}

    if isinstance(data, (list, tuple)):
        return type(data)(detach_tensor_dict(v, to_cpu) for v in data)

    return data


# =========================================================
# MODEL MODES
# =========================================================

def enable_model_eval(model: nn.Module) -> None:
    model.eval()


def enable_model_train(model: nn.Module) -> None:
    model.train()


@contextmanager
def inference_mode():
    with torch.inference_mode():
        yield


# =========================================================
# METRICS CONTAINER (OPTIONAL BUT USEFUL)
# =========================================================

@dataclass
class TrainingMetrics:

    task_losses: Dict[str, float] = field(default_factory=dict)
    losses: Dict[str, float] = field(default_factory=dict)

    step: int = 0
    epoch: int = 0

    def update(self, name: str, value: float) -> None:
        self.losses[name] = float(value)

    def update_task(self, task: str, loss: float) -> None:
        self.task_losses[task] = float(loss)

    def average_loss(self) -> float:
        if not self.task_losses:
            return 0.0
        return sum(self.task_losses.values()) / len(self.task_losses)

    def to_dict(self) -> Dict[str, float]:
        return dict(self.losses)
    
# (Only showing NEW additions + upgrades — your existing code stays)

# =========================================================
# SEED / DETERMINISM
# =========================================================

def set_global_seed(seed: int) -> None:
    import random
    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# =========================================================
# GRADIENT MONITORING
# =========================================================

def compute_grad_norm(model: nn.Module) -> float:
    total_norm = 0.0

    for p in model.parameters():
        if p.grad is None:
            continue

        param_norm = p.grad.data.norm(2)
        total_norm += param_norm.item() ** 2

    return total_norm ** 0.5


# =========================================================
# LR UTILITIES
# =========================================================

def get_current_lr(optimizer: torch.optim.Optimizer) -> float:
    for group in optimizer.param_groups:
        return float(group["lr"])
    return 0.0


# =========================================================
# NAN / INF GUARD
# =========================================================

def check_finite(tensor: torch.Tensor, name: str = "tensor") -> None:
    if not torch.isfinite(tensor).all():
        raise RuntimeError(f"Non-finite values detected in {name}")


# =========================================================
# AMP AUTOCAST WRAPPER
# =========================================================

@contextmanager
def autocast(enabled: bool = True):
    if enabled and torch.cuda.is_available():
        with torch.cuda.amp.autocast():
            yield
    else:
        yield


# =========================================================
# OOM SAFE EXECUTION
# =========================================================

@contextmanager
def safe_cuda_execution():
    try:
        yield
    except RuntimeError as e:
        if "out of memory" in str(e):
            logger.warning("CUDA OOM detected — clearing cache")
            torch.cuda.empty_cache()
        raise


# =========================================================
# THROUGHPUT / TIMING
# =========================================================

class StepTimer:
    def __init__(self):
        self.start_time = None

    def start(self):
        import time
        self.start_time = time.time()

    def stop(self) -> float:
        import time
        return time.time() - self.start_time if self.start_time else 0.0


def compute_throughput(
    batch_size: int,
    duration: float,
) -> float:
    if duration <= 0:
        return 0.0
    return batch_size / duration


# =========================================================
# IMPROVED METRICS CONTAINER
# =========================================================

@dataclass
class TrainingMetrics:

    task_losses: Dict[str, float] = field(default_factory=dict)
    losses: Dict[str, float] = field(default_factory=dict)

    step: int = 0
    epoch: int = 0

    grad_norm: float = 0.0
    lr: float = 0.0
    throughput: float = 0.0

    def update(self, name: str, value: float) -> None:
        self.losses[name] = float(value)

    def update_task(self, task: str, loss: float) -> None:
        self.task_losses[task] = float(loss)

    def set_grad_norm(self, value: float) -> None:
        self.grad_norm = float(value)

    def set_lr(self, value: float) -> None:
        self.lr = float(value)

    def set_throughput(self, value: float) -> None:
        self.throughput = float(value)

    def average_loss(self) -> float:
        if not self.task_losses:
            return 0.0
        return sum(self.task_losses.values()) / len(self.task_losses)

    def to_dict(self) -> Dict[str, float]:
        return {
            **self.losses,
            "grad_norm": self.grad_norm,
            "lr": self.lr,
            "throughput": self.throughput,
        }