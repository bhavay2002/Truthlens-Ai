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
# TASK TYPES
# =========================================================

MULTICLASS_TASKS: frozenset = frozenset({"bias", "ideology", "propaganda"})
MULTILABEL_TASKS: frozenset = frozenset({"narrative", "narrative_frame", "emotion"})


def get_task_type(task: str) -> str:
    if task in MULTICLASS_TASKS:
        return "multiclass"
    if task in MULTILABEL_TASKS:
        return "multilabel"
    raise ValueError(f"Unknown task: {task!r}")


def compute_predictions(task: str, logits: torch.Tensor) -> torch.Tensor:
    if task in MULTICLASS_TASKS:
        return torch.argmax(logits, dim=1)
    if task in MULTILABEL_TASKS:
        return (torch.sigmoid(logits) > 0.5).int()
    raise ValueError(f"Unknown task: {task!r}")


def validate_loss(loss: torch.Tensor) -> None:
    if not torch.isfinite(loss):
        raise RuntimeError(f"Non-finite loss detected: {loss.item()}")
    if loss.item() > 1e4:
        raise RuntimeError(f"Exploding loss detected: {loss.item():.2f}")


# =========================================================
# BATCH VALIDATION
# =========================================================

def extract_task_from_batch(batch: Dict[str, Any]) -> str:
    task = batch.get("task")
    if task is None:
        raise ValueError("Batch missing 'task'")
    if not isinstance(task, str):
        raise TypeError("task must be str")
    return task


def validate_single_task_batch(batch: Dict[str, Any]) -> None:
    required = ("input_ids", "attention_mask", "labels", "task")
    for key in required:
        if key not in batch:
            raise ValueError(f"Missing key: {key}")
    if not isinstance(batch["task"], str):
        raise TypeError("task must be string")


# =========================================================
# METRICS CONTAINER
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
# GRADIENTS
# =========================================================

def clip_gradients(
    parameters: Iterable[nn.Parameter],
    max_norm: Optional[float],
    scaler: Optional[torch.cuda.amp.GradScaler] = None,
) -> float:

    if max_norm is None:
        return 0.0

    if max_norm <= 0:
        raise ValueError("max_norm must be > 0")

    if scaler is not None:
        scaler.unscale_(parameters)

    total_norm = torch.nn.utils.clip_grad_norm_(parameters, max_norm)
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

    raise RuntimeError("Cannot determine batch size")


# =========================================================
# TENSOR UTILS
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