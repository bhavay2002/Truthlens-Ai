"""
File Name: training_step.py
Module: models.training
Description:
    Implements a reusable training step abstraction for TruthLens models.
    The module encapsulates the logic required to execute a single forward
    and backward pass during training, including loss computation, gradient
    accumulation, gradient clipping, optimizer stepping, and scheduler updates.

    This separation allows the Trainer to remain clean and orchestrational
    while the step logic remains modular and testable.

Dependencies:
    logging
    typing
    dataclasses
    torch
    torch.nn
    torch.optim
Inputs:
    Model
    Batch dictionary
Outputs:
    Loss tensor and optional training metrics
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Any

import torch
import torch.nn as nn

from ..checkpointing.checkpoint_manager import CheckpointManager

logger = logging.getLogger(__name__)


@dataclass
class TrainingStepConfig:
    """
    Configuration controlling training step behavior.
    """

    gradient_accumulation_steps: int = 1
    max_grad_norm: float = 1.0
    use_mixed_precision: bool = False
    device: Optional[str] = None
    checkpoint_dir: Optional[str] = None
    checkpoint_every_steps: int = 0
    max_checkpoints: int = 3


class TrainingStep:
    """
    Executes a single training step for a batch.
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[Any],
        config: TrainingStepConfig,
    ) -> None:

        if not isinstance(model, nn.Module):
            raise TypeError("model must be torch.nn.Module")

        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.config = config

        self.device = torch.device(
            config.device if config.device else ("cuda" if torch.cuda.is_available() else "cpu")
        )

        self.scaler = torch.cuda.amp.GradScaler(enabled=config.use_mixed_precision)

        self.checkpoint_manager: Optional[CheckpointManager] = None
        if config.checkpoint_dir:
            self.checkpoint_manager = CheckpointManager(Path(config.checkpoint_dir))

        logger.info("TrainingStep initialized on device %s", self.device)

    def __call__(
        self,
        batch: Dict[str, torch.Tensor],
        step: int,
    ) -> torch.Tensor:
        """
        Execute a single training step.

        Returns:
            Detached loss value.
        """

        batch = self._move_batch_to_device(batch)

        with torch.cuda.amp.autocast(enabled=self.config.use_mixed_precision):

            outputs = self.model(**batch)

            loss = self._extract_loss(outputs)

            loss = loss / self.config.gradient_accumulation_steps

        self.scaler.scale(loss).backward()

        if (step + 1) % self.config.gradient_accumulation_steps == 0:

            if self.config.max_grad_norm is not None:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.max_grad_norm,
                )

            self.scaler.step(self.optimizer)
            self.scaler.update()

            if self.scheduler is not None:
                self.scheduler.step()

            self.optimizer.zero_grad()

            if (
                self.checkpoint_manager is not None
                and self.config.checkpoint_every_steps > 0
                and (step + 1) % self.config.checkpoint_every_steps == 0
            ):
                self.checkpoint_manager.save_checkpoint(
                    step=step + 1,
                    model_state_dict=self.model.state_dict(),
                    optimizer_state_dict=self.optimizer.state_dict(),
                    scheduler_state_dict=(
                        self.scheduler.state_dict() if self.scheduler is not None else None
                    ),
                    metadata={"step_loss": float(loss.detach().item())},
                )
                self.checkpoint_manager.cleanup_old_checkpoints(
                    max_checkpoints=self.config.max_checkpoints
                )

        return loss.detach()

    def _extract_loss(self, outputs: Any) -> torch.Tensor:
        """
        Extract loss tensor from model outputs.
        """

        if isinstance(outputs, dict):

            if "loss" not in outputs:
                raise RuntimeError("Model output dictionary must contain 'loss'")

            return outputs["loss"]

        if hasattr(outputs, "loss"):
            return outputs.loss

        raise RuntimeError("Unable to extract loss from model output")

    def _move_batch_to_device(
        self,
        batch: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """
        Move batch tensors to device.
        """

        moved_batch: Dict[str, torch.Tensor] = {}

        for key, value in batch.items():

            if isinstance(value, torch.Tensor):
                moved_batch[key] = value.to(self.device)
            else:
                moved_batch[key] = value

        return moved_batch