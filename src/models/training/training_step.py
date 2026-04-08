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

import inspect
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Any

import torch
import torch.nn as nn

from ..checkpointing.checkpoint_manager import CheckpointManager
from ..multitask.multitask_output import MultiTaskOutput

logger = logging.getLogger(__name__)

# GPU performance optimization
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


@dataclass
class TrainingStepConfig:

    gradient_accumulation_steps: int = 1
    max_grad_norm: float = 1.0
    use_mixed_precision: bool = False
    device: Optional[str] = None
    checkpoint_dir: Optional[str] = None
    checkpoint_every_steps: int = 0
    max_checkpoints: int = 3


class TrainingStep:

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

        self.model.to(self.device)

        # Mixed precision safety
        self.use_amp = config.use_mixed_precision and torch.cuda.is_available()

        self.scaler = torch.cuda.amp.GradScaler(enabled=self.use_amp)

        # Cache forward signature
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

        self.checkpoint_manager: Optional[CheckpointManager] = None

        if config.checkpoint_dir:
            self.checkpoint_manager = CheckpointManager(Path(config.checkpoint_dir))

        logger.info("TrainingStep initialized on device %s", self.device)

    # ---------------------------------------------------------

    def __call__(
        self,
        batch: Dict[str, torch.Tensor] | tuple | list,
        step: int,
    ) -> torch.Tensor:

        batch = self._move_batch_to_device(batch)

        with torch.cuda.amp.autocast(enabled=self.use_amp):

            outputs = self.model(**self._prepare_model_inputs(batch))

            raw_loss = self._extract_loss(outputs)

            loss = raw_loss / self.config.gradient_accumulation_steps

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

                try:
                    self.scheduler.step()
                except TypeError:
                    self.scheduler.step(raw_loss)

            self.optimizer.zero_grad()

            if (
                self.checkpoint_manager
                and self.config.checkpoint_every_steps > 0
                and (step + 1) % self.config.checkpoint_every_steps == 0
            ):

                self.checkpoint_manager.save_checkpoint(
                    step=step + 1,
                    model_state_dict=self.model.state_dict(),
                    optimizer_state_dict=self.optimizer.state_dict(),
                    scheduler_state_dict=(
                        self.scheduler.state_dict() if self.scheduler else None
                    ),
                    metadata={"step_loss": float(raw_loss.detach().item())},
                )

                self.checkpoint_manager.cleanup_old_checkpoints(
                    max_checkpoints=self.config.max_checkpoints
                )

        return raw_loss.detach()

    # ---------------------------------------------------------

    def _extract_loss(self, outputs: Any) -> torch.Tensor:

        if isinstance(outputs, dict):

            multitask_output = outputs.get("multitask_output")

            if isinstance(multitask_output, MultiTaskOutput):

                if multitask_output.loss is None:
                    raise RuntimeError("MultiTaskOutput exists but loss missing")

                return multitask_output.loss

            if "loss" not in outputs:
                raise RuntimeError("Model output dict must contain 'loss'")

            return outputs["loss"]

        if hasattr(outputs, "loss"):
            return outputs.loss

        raise RuntimeError("Unable to extract loss from model output")

    # ---------------------------------------------------------

    def _move_batch_to_device(self, batch):

        if isinstance(batch, (list, tuple)):
            batch = {"inputs": batch}

        moved_batch: Dict[str, torch.Tensor] = {}

        for key, value in batch.items():

            if isinstance(value, torch.Tensor):
                moved_batch[key] = value.to(self.device)
            else:
                moved_batch[key] = value

        return moved_batch

    # ---------------------------------------------------------

    def _prepare_model_inputs(self, batch: Dict[str, Any]):

        if self._forward_accepts_kwargs:
            return batch

        forward_kwargs: Dict[str, Any] = {}
        label_dict: Dict[str, Any] = {}

        for key, value in batch.items():

            if key in self._forward_params:
                forward_kwargs[key] = value
            else:
                label_dict[key] = value

        if label_dict and "labels" in self._forward_params:

            existing = forward_kwargs.get("labels")

            if isinstance(existing, dict):
                existing.update(label_dict)
            else:
                forward_kwargs["labels"] = label_dict

        return forward_kwargs