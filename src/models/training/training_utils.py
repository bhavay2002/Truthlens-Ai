"""
File Name: training_utils.py
Module: models.training
Description:
    Provides reusable training utilities used across the TruthLens ML training
    pipeline. The module contains helper functions for gradient clipping,
    device management, metric tracking, optimizer stepping, mixed precision
    handling, and reproducible training workflows.

    These utilities help keep trainer implementations clean and reduce
    duplication of common training logic across different training pipelines.

Dependencies:
    logging
    typing
    dataclasses
    torch
    torch.nn
Inputs:
    Model parameters, tensors, metrics, optimizer states
Outputs:
    Utility outputs such as clipped gradients, moved tensors, tracked metrics
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, Iterable, Optional, Any

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

# ---------------------------------------------------------
# GPU PERFORMANCE OPTIMIZATION
# ---------------------------------------------------------

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


@dataclass
class TrainingMetrics:
    """
    Container for tracking training metrics.
    """

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
) -> Any:
    """
    Move tensors to device recursively.
    """

    if isinstance(batch, torch.Tensor):
        return batch.to(device)

    if isinstance(batch, dict):
        return {k: move_batch_to_device(v, device) for k, v in batch.items()}

    if isinstance(batch, (list, tuple)):
        return type(batch)(move_batch_to_device(v, device) for v in batch)

    return batch


# ---------------------------------------------------------
# GRADIENT UTILITIES
# ---------------------------------------------------------

def clip_gradients(
    parameters: Iterable[nn.Parameter],
    max_norm: Optional[float],
) -> float:
    """
    Clip gradients to stabilize training.
    """

    if max_norm is None:
        return 0.0

    if max_norm <= 0:
        raise ValueError("max_norm must be positive")

    total_norm = torch.nn.utils.clip_grad_norm_(parameters, max_norm)

    return float(total_norm)


def zero_gradients(model: nn.Module) -> None:
    """
    Efficiently zero gradients.
    """

    model.zero_grad(set_to_none=True)


# ---------------------------------------------------------
# BATCH UTILITIES
# ---------------------------------------------------------

def compute_batch_size(batch: Any) -> int:
    """
    Infer batch size from batch structure.
    """

    if isinstance(batch, torch.Tensor):

        if batch.ndim == 0:
            return 1

        return batch.size(0)

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

def detach_tensor_dict(
    tensor_dict: Any,
) -> Any:
    """
    Detach tensors recursively for logging.
    """

    if isinstance(tensor_dict, torch.Tensor):
        return tensor_dict.detach().cpu()

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