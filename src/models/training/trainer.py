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

import inspect
import logging
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Any, List

import torch
from torch import nn
from torch.utils.data import DataLoader

from ..checkpointing.checkpoint_manager import CheckpointManager
from src.training.checkpointing import (
    list_checkpoints as list_training_checkpoints,
    resume_training as resume_training_checkpoint,
    save_checkpoint as save_training_checkpoint,
)
from src.utils import create_folder, get_device, move_to_device

logger = logging.getLogger(__name__)

# ---------------------------------------------------------
# GPU PERFORMANCE SETTINGS
# ---------------------------------------------------------

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

torch.set_float32_matmul_precision("high")


@dataclass
class TrainerConfig:

    epochs: int = 3
    gradient_accumulation_steps: int = 1
    max_grad_norm: float = 1.0
    device: Optional[str] = None
    log_every_steps: int = 50
    checkpoint_dir: Optional[str] = None
    drive_checkpoint_dir: Optional[str] = None
    checkpoint_every_steps: int = 0
    max_checkpoints: int = 3


class Trainer:

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[Any],
        config: TrainerConfig,
    ) -> None:

        if not isinstance(model, nn.Module):
            raise TypeError("model must be torch.nn.Module")

        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.config = config

        self.device = (
            torch.device(config.device)
            if config.device
            else get_device(prefer_gpu=True)
        )

        self.model.to(self.device)

        # AMP (Mixed Precision)
        self.use_amp = torch.cuda.is_available()
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.use_amp)

        # Forward signature caching
        try:
            sig = inspect.signature(self.model.forward)
            self._forward_params = set(sig.parameters.keys()) - {"self"}
            self._forward_accepts_kwargs = any(
                p.kind == inspect.Parameter.VAR_KEYWORD
                for p in sig.parameters.values()
            )
        except Exception:
            self._forward_params = None
            self._forward_accepts_kwargs = True

        self.global_step = 0

        self.checkpoint_manager: Optional[CheckpointManager] = None

        if config.checkpoint_dir:
            self.checkpoint_manager = CheckpointManager(Path(config.checkpoint_dir))
            self._attempt_resume()

        logger.info("Trainer initialized on device %s", self.device)

    # ---------------------------------------------------------

    def _attempt_resume(self):

        if not self.config.checkpoint_dir:
            return

        checkpoint_root = Path(self.config.checkpoint_dir)

        available = list_training_checkpoints(checkpoint_root)

        if not available:
            return

        latest = available[-1]

        try:

            state = resume_training_checkpoint(
                self.model,
                checkpoint_dir=latest,
                optimizer=self.optimizer,
                scheduler=self.scheduler,
                map_location=self.device,
            )

            self.global_step = int(state.get("start_step", 0) or 0)

            logger.info("Resumed training from %s", latest)

        except Exception as exc:
            logger.warning("Checkpoint resume skipped: %s", exc)

    # ---------------------------------------------------------

    def train(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
    ) -> Dict[str, List[float]]:

        history = {"train_loss": [], "val_loss": []}

        for epoch in range(self.config.epochs):

            logger.info("Epoch %d/%d", epoch + 1, self.config.epochs)

            train_loss = self._train_epoch(train_loader, epoch)

            history["train_loss"].append(train_loss)

            if val_loader:

                val_loss = self._validate_epoch(val_loader)

                history["val_loss"].append(val_loss)

        return history

    # ---------------------------------------------------------

    def _train_epoch(self, dataloader: DataLoader, epoch: int) -> float:

        self.model.train()

        total_loss = 0.0
        step_count = 0

        self.optimizer.zero_grad()

        for step, batch in enumerate(dataloader):

            batch = self._move_batch_to_device(batch)

            with torch.cuda.amp.autocast(enabled=self.use_amp):

                outputs = self.model(**self._prepare_model_inputs(batch))

                raw_loss = self._extract_loss(outputs)

                loss = raw_loss / self.config.gradient_accumulation_steps

            self.scaler.scale(loss).backward()

            total_loss += raw_loss.item()

            step_count += 1
            self.global_step += 1

            if (step + 1) % self.config.gradient_accumulation_steps == 0:

                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.max_grad_norm,
                )

                self.scaler.step(self.optimizer)
                self.scaler.update()

                if self.scheduler:
                    try:
                        self.scheduler.step()
                    except TypeError:
                        self.scheduler.step(raw_loss)

                self.optimizer.zero_grad()

            if (step + 1) % self.config.log_every_steps == 0:

                logger.info(
                    "step %d | loss %.6f",
                    step + 1,
                    raw_loss.item(),
                )

        # FINAL GRADIENT STEP FIX
        if step_count % self.config.gradient_accumulation_steps != 0:

            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.config.max_grad_norm,
            )

            self.scaler.step(self.optimizer)
            self.scaler.update()

            if self.scheduler:
                self.scheduler.step()

            self.optimizer.zero_grad()

        return total_loss / max(step_count, 1)

    # ---------------------------------------------------------

    def _validate_epoch(self, dataloader: DataLoader) -> float:

        self.model.eval()

        total_loss = 0.0
        step_count = 0

        with torch.no_grad():

            for batch in dataloader:

                batch = self._move_batch_to_device(batch)

                outputs = self.model(**self._prepare_model_inputs(batch))

                loss = self._extract_loss(outputs)

                total_loss += loss.item()

                step_count += 1

        return total_loss / max(step_count, 1)

    # ---------------------------------------------------------

    def _extract_loss(self, outputs):

        if isinstance(outputs, dict):

            if "loss" not in outputs:
                raise RuntimeError("Model output must contain 'loss'")

            return outputs["loss"]

        if hasattr(outputs, "loss"):
            return outputs.loss

        raise RuntimeError("Unable to extract loss")

    # ---------------------------------------------------------

    def _move_batch_to_device(self, batch):

        if isinstance(batch, dict):
            return move_to_device(batch, self.device)

        if isinstance(batch, (list, tuple)):
            return move_to_device({"inputs": batch}, self.device)

        raise TypeError("Unsupported batch format")

    # ---------------------------------------------------------

    def _prepare_model_inputs(self, batch):

        if self._forward_accepts_kwargs:
            return batch

        forward_kwargs = {}
        label_dict = {}

        for key, value in batch.items():

            if key in self._forward_params:
                forward_kwargs[key] = value
            else:
                label_dict[key] = value

        if label_dict and "labels" in self._forward_params:
            forward_kwargs["labels"] = label_dict

        return forward_kwargs