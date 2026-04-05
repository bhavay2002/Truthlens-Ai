"""
File Name: trainer.py
Module: models.training
Description:
    Implements the training engine for TruthLens models. This module provides a
    reusable Trainer abstraction responsible for coordinating the full training
    lifecycle including forward passes, backpropagation, gradient accumulation,
    optimizer steps, scheduler updates, checkpointing hooks, and metric logging.

    The trainer is framework-agnostic with respect to the model architecture and
    supports both single-task and multi-task models that return either dictionaries
    or structured outputs.

    Designed for research reproducibility and production ML pipelines.

Dependencies:
    logging
    typing
    dataclasses
    torch
    torch.nn
    torch.optim
Inputs:
    Model
    Training DataLoader
    Validation DataLoader
Outputs:
    Training history and trained model parameters
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Any, List

import torch
from torch import nn
from torch.utils.data import DataLoader

from ..checkpointing.checkpoint_manager import CheckpointManager

logger = logging.getLogger(__name__)


@dataclass
class TrainerConfig:
    """
    Configuration for the training process.
    """

    epochs: int = 3
    gradient_accumulation_steps: int = 1
    max_grad_norm: float = 1.0
    device: Optional[str] = None
    log_every_steps: int = 50
    checkpoint_dir: Optional[str] = None
    checkpoint_every_steps: int = 0
    max_checkpoints: int = 3


class Trainer:
    """
    Generic trainer for TruthLens models.
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[Any],
        config: TrainerConfig,
    ) -> None:

        if not isinstance(model, nn.Module):
            raise TypeError("model must be a torch.nn.Module")

        if not isinstance(optimizer, torch.optim.Optimizer):
            raise TypeError("optimizer must be a torch optimizer")

        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.config = config

        self.device = torch.device(
            config.device if config.device else ("cuda" if torch.cuda.is_available() else "cpu")
        )

        self.model.to(self.device)
        self.global_step = 0

        self.checkpoint_manager: Optional[CheckpointManager] = None
        if config.checkpoint_dir:
            self.checkpoint_manager = CheckpointManager(Path(config.checkpoint_dir))

        logger.info("Trainer initialized on device %s", self.device)

    def train(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
    ) -> Dict[str, List[float]]:
        """
        Execute full training loop.
        """

        history: Dict[str, List[float]] = {
            "train_loss": [],
            "val_loss": [],
        }

        for epoch in range(self.config.epochs):

            logger.info("Starting epoch %d/%d", epoch + 1, self.config.epochs)

            train_loss = self._train_epoch(train_loader, epoch)

            history["train_loss"].append(train_loss)

            logger.info("Epoch %d training loss: %.6f", epoch + 1, train_loss)

            if self.checkpoint_manager is not None:
                self.checkpoint_manager.save_checkpoint(
                    step=self.global_step,
                    model_state_dict=self.model.state_dict(),
                    optimizer_state_dict=self.optimizer.state_dict(),
                    scheduler_state_dict=(
                        self.scheduler.state_dict() if self.scheduler is not None else None
                    ),
                    metadata={
                        "epoch": epoch + 1,
                        "train_loss": float(train_loss),
                    },
                )
                self.checkpoint_manager.cleanup_old_checkpoints(
                    max_checkpoints=self.config.max_checkpoints
                )

            if val_loader is not None:

                val_loss = self._validate_epoch(val_loader)

                history["val_loss"].append(val_loss)

                logger.info("Epoch %d validation loss: %.6f", epoch + 1, val_loss)

        return history

    def _train_epoch(self, dataloader: DataLoader, epoch: int) -> float:
        """
        Train model for a single epoch.
        """

        self.model.train()

        total_loss = 0.0
        step_count = 0

        self.optimizer.zero_grad()

        for step, batch in enumerate(dataloader):

            batch = self._move_batch_to_device(batch)

            outputs = self.model(**batch)

            loss = self._extract_loss(outputs)

            loss = loss / self.config.gradient_accumulation_steps

            loss.backward()

            total_loss += loss.item()
            step_count += 1
            self.global_step += 1

            if (step + 1) % self.config.gradient_accumulation_steps == 0:

                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.max_grad_norm,
                )

                self.optimizer.step()

                if self.scheduler is not None:
                    self.scheduler.step()

                self.optimizer.zero_grad()

            if (
                self.checkpoint_manager is not None
                and self.config.checkpoint_every_steps > 0
                and self.global_step % self.config.checkpoint_every_steps == 0
            ):
                self.checkpoint_manager.save_checkpoint(
                    step=self.global_step,
                    model_state_dict=self.model.state_dict(),
                    optimizer_state_dict=self.optimizer.state_dict(),
                    scheduler_state_dict=(
                        self.scheduler.state_dict() if self.scheduler is not None else None
                    ),
                    metadata={
                        "epoch": epoch + 1,
                        "step_loss": float(loss.item()),
                    },
                )
                self.checkpoint_manager.cleanup_old_checkpoints(
                    max_checkpoints=self.config.max_checkpoints
                )

            if (step + 1) % self.config.log_every_steps == 0:
                logger.info(
                    "Training step %d | loss %.6f",
                    step + 1,
                    loss.item(),
                )

        avg_loss = total_loss / max(step_count, 1)

        return avg_loss

    def _validate_epoch(self, dataloader: DataLoader) -> float:
        """
        Run validation loop.
        """

        self.model.eval()

        total_loss = 0.0
        step_count = 0

        with torch.no_grad():

            for batch in dataloader:

                batch = self._move_batch_to_device(batch)

                outputs = self.model(**batch)

                loss = self._extract_loss(outputs)

                total_loss += loss.item()
                step_count += 1

        avg_loss = total_loss / max(step_count, 1)

        return avg_loss

    def _extract_loss(self, outputs: Any) -> torch.Tensor:
        """
        Extract loss tensor from model output.
        """

        if isinstance(outputs, dict):

            if "loss" not in outputs:
                raise RuntimeError("Model output dictionary must contain 'loss'")

            return outputs["loss"]

        if hasattr(outputs, "loss"):
            return outputs.loss

        raise RuntimeError("Unable to extract loss from model output")

    def _move_batch_to_device(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Move batch tensors to the configured device.
        """

        if not isinstance(batch, dict):
            raise TypeError("Batch must be a dictionary")

        moved_batch: Dict[str, torch.Tensor] = {}

        for key, value in batch.items():

            if isinstance(value, torch.Tensor):
                moved_batch[key] = value.to(self.device)
            else:
                moved_batch[key] = value

        return moved_batch