"""
File Name: base_model.py
Module: models.base
Description:
    Defines the abstract base class for all models in the TruthLens ML framework.
    This module establishes a consistent interface for training, inference, and
    checkpoint management across all model implementations. It provides common
    utilities for device management, forward execution, model saving/loading,
    and parameter inspection. Designed for compatibility with PyTorch-based
    architectures and extensibility for research and production deployments.

Dependencies:
    torch
    torch.nn
    logging
    pathlib
    typing
Inputs:
    Model inputs (tensor batches) defined by child classes.
Outputs:
    Model outputs defined by child implementations.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, Optional

import torch
from torch import nn

logger = logging.getLogger(__name__)


class BaseModel(nn.Module, ABC):
    """
    Abstract base class for all TruthLens models.

    This class defines a consistent interface for:
    - forward passes
    - model checkpointing
    - device management
    - parameter inspection

    All task-specific models and multitask models should inherit from this class.
    """

    def __init__(self) -> None:
        super().__init__()
        self._device: torch.device = torch.device("cpu")

    @abstractmethod
    def forward(self, *inputs: torch.Tensor, **kwargs: Any) -> Any:
        """
        Executes a forward pass of the model.

        Args:
            *inputs: Positional tensor inputs.
            **kwargs: Additional named inputs.

        Returns:
            Model-specific outputs.
        """
        raise NotImplementedError("Forward method must be implemented by subclass.")

    def set_device(self, device: torch.device) -> None:
        """
        Moves model parameters to a specified device.

        Args:
            device: Target device (CPU or CUDA).
        """
        if not isinstance(device, torch.device):
            raise TypeError("device must be an instance of torch.device")

        logger.info("Moving model to device: %s", device)
        self._device = device
        self.to(device)

    @property
    def device(self) -> torch.device:
        """
        Returns the device where the model resides.
        """
        return self._device

    def num_parameters(self, trainable_only: bool = True) -> int:
        """
        Computes the number of model parameters.

        Args:
            trainable_only: If True, count only trainable parameters.

        Returns:
            Total number of parameters.
        """
        if trainable_only:
            params = (p for p in self.parameters() if p.requires_grad)
        else:
            params = self.parameters()

        count = sum(p.numel() for p in params)
        logger.debug("Parameter count (trainable_only=%s): %d", trainable_only, count)
        return count

    def save_checkpoint(
        self,
        path: Path,
        optimizer_state: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Saves model checkpoint.

        Args:
            path: Destination checkpoint path.
            optimizer_state: Optional optimizer state dictionary.
            metadata: Optional experiment metadata.
        """
        if not isinstance(path, Path):
            raise TypeError("path must be a pathlib.Path instance")

        path.parent.mkdir(parents=True, exist_ok=True)

        checkpoint = {
            "model_state_dict": self.state_dict(),
            "optimizer_state_dict": optimizer_state,
            "metadata": metadata,
        }

        try:
            torch.save(checkpoint, path)
            logger.info("Checkpoint saved to %s", path)
        except Exception as exc:
            logger.exception("Failed to save checkpoint: %s", exc)
            raise

    def load_checkpoint(
        self,
        path: Path,
        optimizer: Optional[torch.optim.Optimizer] = None,
        map_location: Optional[str | torch.device] = None,
    ) -> Dict[str, Any]:
        """
        Loads a model checkpoint.

        Args:
            path: Path to checkpoint file.
            optimizer: Optional optimizer to restore state.
            map_location: Device mapping for loading.

        Returns:
            Metadata dictionary stored with checkpoint.
        """
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {path}")

        try:
            checkpoint = torch.load(path, map_location=map_location, weights_only=False)
            self.load_state_dict(checkpoint["model_state_dict"])

            if optimizer and checkpoint.get("optimizer_state_dict"):
                optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

            logger.info("Checkpoint loaded from %s", path)

            return checkpoint.get("metadata", {})

        except Exception as exc:
            logger.exception("Failed to load checkpoint: %s", exc)
            raise

    def freeze(self) -> None:
        """
        Freezes all model parameters.
        """
        for param in self.parameters():
            param.requires_grad = False

        logger.info("All model parameters frozen.")

    def unfreeze(self) -> None:
        """
        Unfreezes all model parameters.
        """
        for param in self.parameters():
            param.requires_grad = True

        logger.info("All model parameters unfrozen.")

    def summary(self) -> Dict[str, Any]:
        """
        Returns a summary of model properties.

        Returns:
            Dictionary containing model metadata.
        """
        summary_data = {
            "model_class": self.__class__.__name__,
            "device": str(self.device),
            "trainable_parameters": self.num_parameters(True),
            "total_parameters": self.num_parameters(False),
        }

        logger.debug("Model summary: %s", summary_data)
        return summary_data