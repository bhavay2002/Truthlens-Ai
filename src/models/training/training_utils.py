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

@dataclass
class TrainingMetrics:
    losses: Dict[str, float] = field(default_factory=dict)
    step: int = 0
    epoch: int = 0

    def update(self, name: str, value: float) -> None:
        if not isinstance(name, str):
            raise TypeError("Metric name must be a string")
        if not isinstance(value, (float, int)):
            raise TypeError("Metric value must be numeric")

        self.losses[name] = float(value)

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

    if isinstance(batch, torch.Tensor):
        if batch.ndim == 0:
            return 1
        return batch.shape[0]

    if isinstance(batch, dict):
        for value in batch.values():
            size = compute_batch_size(value)
            if size > 0:
                return size

    if isinstance(batch, (list, tuple)):
        for value in batch:
            size = compute_batch_size(value)
            if size > 0:
                return size

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