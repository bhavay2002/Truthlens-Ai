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
from typing import Dict, Iterable, Optional

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


@dataclass
class TrainingMetrics:
    """
    Container for tracking training metrics.
    """

    losses: Dict[str, float] = field(default_factory=dict)
    step: int = 0
    epoch: int = 0

    def update(self, name: str, value: float) -> None:
        """
        Update metric value.
        """

        if not isinstance(name, str):
            raise TypeError("Metric name must be a string")

        self.losses[name] = value

    def to_dict(self) -> Dict[str, float]:
        """
        Return metrics dictionary.
        """

        return dict(self.losses)


def move_batch_to_device(
    batch: Dict[str, torch.Tensor],
    device: torch.device,
) -> Dict[str, torch.Tensor]:
    """
    Move batch tensors to target device.
    """

    if not isinstance(batch, dict):
        raise TypeError("Batch must be a dictionary")

    moved_batch: Dict[str, torch.Tensor] = {}

    for key, value in batch.items():

        if isinstance(value, torch.Tensor):
            moved_batch[key] = value.to(device)
        else:
            moved_batch[key] = value

    return moved_batch


def clip_gradients(
    parameters: Iterable[torch.nn.Parameter],
    max_norm: float,
) -> float:
    """
    Clip gradients to stabilize training.
    """

    if max_norm <= 0:
        raise ValueError("max_norm must be positive")

    total_norm = torch.nn.utils.clip_grad_norm_(parameters, max_norm)

    return float(total_norm)


def get_device(device: Optional[str] = None) -> torch.device:
    """
    Resolve device automatically if not provided.
    """

    if device:
        return torch.device(device)

    resolved = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    logger.debug("Resolved training device: %s", resolved)

    return resolved


def zero_gradients(model: nn.Module) -> None:
    """
    Zero model gradients.
    """

    for param in model.parameters():
        if param.grad is not None:
            param.grad.detach_()
            param.grad.zero_()


def compute_batch_size(batch: Dict[str, torch.Tensor]) -> int:
    """
    Infer batch size from input batch dictionary.
    """

    for value in batch.values():
        if isinstance(value, torch.Tensor):
            return value.size(0)

    raise RuntimeError("Unable to determine batch size from batch")


def detach_tensor_dict(
    tensor_dict: Dict[str, torch.Tensor],
) -> Dict[str, torch.Tensor]:
    """
    Detach tensors for logging without affecting computation graph.
    """

    detached: Dict[str, torch.Tensor] = {}

    for key, value in tensor_dict.items():

        if isinstance(value, torch.Tensor):
            detached[key] = value.detach().cpu()
        else:
            detached[key] = value

    return detached


def enable_model_eval(model: nn.Module) -> None:
    """
    Set model to evaluation mode.
    """

    model.eval()


def enable_model_train(model: nn.Module) -> None:
    """
    Set model to training mode.
    """

    model.train()