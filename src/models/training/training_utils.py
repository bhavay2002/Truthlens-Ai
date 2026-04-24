from __future__ import annotations

import logging
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Dict, Iterable, Optional, Any

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

def configure_training_precision(
    *,
    allow_tf32: bool = True,
    matmul_precision: str = "high",
) -> None:
    """
    Configure precision settings for training.

    This function must be called explicitly at the start of a training run.
    It does not execute on import to avoid global side effects in inference.
    """
    if torch.cuda.is_available():
        try:
            torch.backends.cuda.matmul.allow_tf32 = bool(allow_tf32)
            torch.backends.cudnn.allow_tf32 = bool(allow_tf32)
        except Exception as exc:
            logger.debug("TF32 configuration skipped: %s", exc)

    try:
        torch.set_float32_matmul_precision(matmul_precision)
    except Exception as exc:
        logger.debug("Matmul precision configuration skipped: %s", exc)


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


# ---------------------------------------------------------
# METRICS
# ---------------------------------------------------------

MULTICLASS_TASKS: frozenset = frozenset({"bias", "ideology", "propaganda"})
MULTILABEL_TASKS: frozenset = frozenset({"narrative", "narrative_frame", "emotion"})


def get_task_type(task: str) -> str:
    """Return 'multiclass' or 'multilabel' for a known task name."""
    if task in MULTICLASS_TASKS:
        return "multiclass"
    if task in MULTILABEL_TASKS:
        return "multilabel"
    raise ValueError(f"Unknown task: {task!r}")


def compute_predictions(task: str, logits: torch.Tensor) -> torch.Tensor:
    """Return argmax (multiclass) or sigmoid>0.5 (multilabel) predictions."""
    if task in MULTICLASS_TASKS:
        return torch.argmax(logits, dim=1)
    if task in MULTILABEL_TASKS:
        return (torch.sigmoid(logits) > 0.5).int()
    raise ValueError(f"Unknown task: {task!r}")


def validate_loss(loss: torch.Tensor) -> None:
    """Raise if loss is non-finite or exploding."""
    if not torch.isfinite(loss):
        raise RuntimeError(f"Non-finite loss detected: {loss.item()}")
    if loss.item() > 1e4:
        raise RuntimeError(f"Exploding loss detected: {loss.item():.2f}")


def extract_task_from_batch(batch: Dict[str, Any]) -> str:
    """Extract the 'task' string from a batch dict. Raises if missing."""
    task = batch.get("task")
    if task is None:
        raise ValueError("Batch missing required 'task' field")
    if not isinstance(task, str):
        raise TypeError(f"batch['task'] must be str, got {type(task)}")
    return task


def validate_single_task_batch(batch: Dict[str, Any]) -> None:
    """Assert that a batch has the required single-task structure."""
    required = ("input_ids", "attention_mask", "labels", "task")
    for key in required:
        if key not in batch:
            raise ValueError(f"Batch missing required key: {key!r}")
    if not isinstance(batch["task"], str):
        raise TypeError("batch['task'] must be a string")


@dataclass
class TrainingMetrics:
    task_losses: Dict[str, float] = field(default_factory=dict)
    losses: Dict[str, float] = field(default_factory=dict)
    step: int = 0
    epoch: int = 0

    def update(self, name: str, value: float) -> None:
        if not isinstance(name, str):
            raise TypeError("Metric name must be a string")
        if not isinstance(value, (float, int)):
            raise TypeError("Metric value must be numeric")
        self.losses[name] = float(value)

    def update_task(self, task: str, loss: float) -> None:
        self.task_losses[task] = float(loss)

    def average_loss(self) -> float:
        if not self.task_losses:
            return 0.0
        return sum(self.task_losses.values()) / len(self.task_losses)

    def to_dict(self) -> Dict[str, float]:
        return dict(self.losses)


# ---------------------------------------------------------
# DEVICE UTILITIES
# ---------------------------------------------------------

def get_device(device: Optional[str] = None) -> torch.device:
    if device:
        return torch.device(device)

    resolved = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.debug("Resolved training device: %s", resolved)
    return resolved


def move_batch_to_device(
    batch: Any,
    device: torch.device,
    non_blocking: bool = True,
) -> Any:
    """
    Move tensors to device recursively (optimized for pinned memory).
    """

    if isinstance(batch, torch.Tensor):
        use_non_blocking = (
            non_blocking
            and batch.device.type == "cpu"
            and device.type == "cuda"
            and batch.is_pinned()
        )
        return batch.to(device, non_blocking=use_non_blocking)

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


# ---------------------------------------------------------
# GRADIENT UTILITIES
# ---------------------------------------------------------

def clip_gradients(
    parameters: Iterable[nn.Parameter],
    max_norm: Optional[float],
    scaler: Optional[torch.cuda.amp.GradScaler] = None,
) -> float:

    if max_norm is None:
        return 0.0

    if max_norm <= 0:
        raise ValueError("max_norm must be positive")

    if scaler is not None:
        scaler.unscale_(parameters)

    total_norm = torch.nn.utils.clip_grad_norm_(parameters, max_norm)
    return float(total_norm)


def zero_gradients(optimizer: torch.optim.Optimizer) -> None:
    """
    Faster zero grad (preferred over model.zero_grad)
    """
    optimizer.zero_grad(set_to_none=True)


# ---------------------------------------------------------
# BATCH UTILITIES
# ---------------------------------------------------------

def compute_batch_size(batch: Any) -> int:

    if isinstance(batch, dict) and "input_ids" in batch:
        return batch["input_ids"].shape[0]

    if isinstance(batch, torch.Tensor):
        if batch.ndim == 0:
            return 1
        return batch.shape[0]

    if isinstance(batch, dict):
        for value in batch.values():
            if isinstance(value, torch.Tensor) and value.ndim > 0:
                return value.shape[0]

    if isinstance(batch, (list, tuple)):
        for value in batch:
            if isinstance(value, torch.Tensor) and value.ndim > 0:
                return value.shape[0]

    raise RuntimeError("Unable to determine batch size")


# ---------------------------------------------------------
# TENSOR UTILITIES
# ---------------------------------------------------------

def detach_tensor_dict(tensor_dict: Any, to_cpu: bool = True) -> Any:

    if isinstance(tensor_dict, torch.Tensor):
        detached = tensor_dict.detach()
        return detached.cpu() if to_cpu else detached

    if isinstance(tensor_dict, dict):
        return {k: detach_tensor_dict(v) for k, v in tensor_dict.items()}

    if isinstance(tensor_dict, (list, tuple)):
        return type(tensor_dict)(detach_tensor_dict(v) for v in tensor_dict)

    return tensor_dict


# ---------------------------------------------------------
# MODEL MODE UTILITIES
# ---------------------------------------------------------

def enable_model_eval(model: nn.Module) -> None:
    model.eval()


def enable_model_train(model: nn.Module) -> None:
    model.train()


@contextmanager
def inference_mode():
    with torch.inference_mode():
        yield