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
import os
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
)
from src.utils import get_device, move_to_device

logger = logging.getLogger(__name__)


def _configure_tf32() -> None:
    """Enable TF32 + FP16 reduced-precision reduction when CUDA is available.

    Invoked inside Trainer.__init__ so importing this module has no global
    side effects on numerical precision.
    """
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision("high")
        # Tensor Core friendly FP16 reductions (no measurable accuracy loss
        # for transformer training).
        if hasattr(torch.backends.cuda.matmul, "allow_fp16_reduced_precision_reduction"):
            torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = True


# ---------------------------------------------------------
# UTIL
# ---------------------------------------------------------

def _get_autocast_dtype():
    if torch.cuda.is_available():
        if torch.cuda.is_bf16_supported():
            return torch.bfloat16
        return torch.float16
    return torch.float32


# ---------------------------------------------------------
# CONFIG
# ---------------------------------------------------------

@dataclass
class TrainerConfig:
    epochs: int = 3
    gradient_accumulation_steps: int = 1
    max_grad_norm: float = 1.0
    device: Optional[str] = None
    log_every_steps: int = 100
    checkpoint_dir: Optional[str] = None
    checkpoint_every_steps: int = 0
    use_amp: Optional[bool] = None
    amp_dtype: Optional[str] = None
    # Run validation every N epochs (default 2). Saves 10-20% wall time.
    validate_every_n_epochs: int = 1


# ---------------------------------------------------------
# TRAINER
# ---------------------------------------------------------

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

        _configure_tf32()

        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.config = config

        # Device
        self.device = (
            torch.device(config.device)
            if config.device
            else get_device(prefer_gpu=True)
        )

        self.model.to(self.device)

        #  torch.compile (CUDA-only, idempotent — C2)
        # Enabled by default on CUDA: pays off ~10-20% on Ampere+ (A100/L4/H100).
        # Disable on T4 (or for debugging) with TRUTHLENS_TORCH_COMPILE=0.
        if (
            os.environ.get("TRUTHLENS_TORCH_COMPILE", "1") == "1"
            and hasattr(torch, "compile")
            and self.device.type == "cuda"
            and not getattr(self.model, "_dynamo_compiled", False)
        ):
            try:
                self.model = torch.compile(self.model)
                try:
                    self.model._dynamo_compiled = True
                except Exception:
                    pass
                logger.info("torch.compile enabled")
            except Exception as e:
                logger.warning(f"torch.compile failed: {e}")

        # AMP Setup
        if self.config.use_amp is None:
            self.use_amp = self.device.type == "cuda"
        else:
            self.use_amp = bool(self.config.use_amp)

        if self.config.amp_dtype:
            if self.config.amp_dtype.lower() == "bf16":
                self.autocast_dtype = torch.bfloat16
            elif self.config.amp_dtype.lower() == "fp16":
                self.autocast_dtype = torch.float16
            else:
                self.autocast_dtype = _get_autocast_dtype()
        else:
            self.autocast_dtype = _get_autocast_dtype()
        if self.device.type == "cuda":
            self.autocast_device_type = "cuda"
        else:
            self.autocast_device_type = "cpu"
            self.use_amp = False

        self.scaler = torch.amp.GradScaler(
            "cuda",
            enabled=self.use_amp and self.autocast_dtype == torch.float16
        )

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

        # Checkpoint manager
        self.checkpoint_manager: Optional[CheckpointManager] = None
        if config.checkpoint_dir:
            self.checkpoint_manager = CheckpointManager(Path(config.checkpoint_dir))
            self._attempt_resume()

        logger.info("Trainer initialized on device %s", self.device)

    # ---------------------------------------------------------

    def _attempt_resume(self):

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
            # C4: scheduler state is restored inside resume_training_checkpoint.
            # The previously-duplicated branch here read keys that resume_training
            # never returned and was dead code — removed.
            logger.info("Resumed training from %s", latest)

        except Exception as exc:
            logger.warning("Checkpoint resume skipped: %s", exc)

    # ---------------------------------------------------------

    def train(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
    ) -> Dict[str, List[float]]:

        history: Dict[str, List[float]] = {"train_loss": [], "val_loss": []}
        best_val = float("inf")

        validate_every = max(1, int(getattr(self.config, "validate_every_n_epochs", 1)))

        for epoch in range(self.config.epochs):

            logger.info("Epoch %d/%d", epoch + 1, self.config.epochs)

            train_loss = self._train_epoch(train_loader)
            history["train_loss"].append(train_loss)

            val_loss: Optional[float] = None
            is_last_epoch = (epoch + 1) == self.config.epochs
            should_validate = (
                val_loader is not None
                and (((epoch + 1) % validate_every == 0) or is_last_epoch)
            )
            if should_validate:
                val_loss = self._validate_epoch(val_loader)
                history["val_loss"].append(val_loss)

            # C3: epoch-level checkpointing + best-model marker
            if self.checkpoint_manager is not None:
                metadata = {
                    "epoch": epoch + 1,
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                }
                try:
                    self.checkpoint_manager.save_checkpoint(
                        step=self.global_step,
                        model=self.model,
                        optimizer=self.optimizer,
                        scheduler=self.scheduler,
                        metadata=metadata,
                        save_optimizer=True,
                        save_every=1,
                        deduplicate=False,
                    )
                    if val_loss is not None and val_loss < best_val:
                        best_val = val_loss
                        self.checkpoint_manager.save_checkpoint(
                            step=10**9 + epoch,
                            model=self.model,
                            metadata={**metadata, "marker": "best"},
                            save_every=1,
                            deduplicate=False,
                        )
                    self.checkpoint_manager.cleanup_old_checkpoints(max_checkpoints=3)
                except Exception as exc:
                    logger.error(
                        "Checkpoint save failed at epoch %d: %s",
                        epoch + 1, exc, exc_info=True,
                    )

        return history

    # ---------------------------------------------------------

    def _train_epoch(self, dataloader: DataLoader) -> float:

        self.model.train()

        # Accumulate loss on-device to avoid per-step GPU→CPU sync.
        loss_accum = torch.zeros((), device=self.device, dtype=torch.float32)
        step_count = 0

        self.optimizer.zero_grad(set_to_none=True)

        step = -1  # M3: bind step in case dataloader is empty
        for step, batch in enumerate(dataloader):

            batch = self._move_batch_to_device(batch)

            with torch.autocast(
                device_type=self.autocast_device_type,
                dtype=self.autocast_dtype,
                enabled=self.use_amp,
            ):
                outputs = self.model(**self._prepare_model_inputs(batch))
                raw_loss = self._extract_loss(outputs)
                loss = raw_loss / self.config.gradient_accumulation_steps

            if not torch.isfinite(raw_loss):
                # M6: poisoned grads from the in-progress accumulation window
                # would otherwise leak into the next optimizer.step. Reset.
                logger.error(
                    "NaN/Inf loss at step %d — resetting accumulation", step
                )
                self.optimizer.zero_grad(set_to_none=True)
                continue

            if self.scaler.is_enabled():
                self.scaler.scale(loss).backward()
            else:
                loss.backward()

            loss_accum = loss_accum + raw_loss.detach().to(loss_accum.dtype)
            step_count += 1
            self.global_step += 1

            # -------------------------------------------------
            # STEP CHECKPOINTING (every N global steps)
            # -------------------------------------------------
            if (
                self.checkpoint_manager is not None
                and self.config.checkpoint_every_steps > 0
                and self.global_step % self.config.checkpoint_every_steps == 0
            ):
                try:
                    self.checkpoint_manager.save_checkpoint(
                        step=self.global_step,
                        model=self.model,
                        optimizer=self.optimizer,
                        scheduler=self.scheduler,
                        metadata={
                            "step": self.global_step,
                            "epoch": None,
                            "type": "step",
                        },
                        save_optimizer=True,
                        save_every=1,
                        deduplicate=False,
                    )
                    logger.info("[Checkpoint] Saved at step %d", self.global_step)
                    self.checkpoint_manager.cleanup_old_checkpoints(max_checkpoints=3)
                except Exception as exc:
                    logger.error(
                        "[Checkpoint] Failed at step %d: %s",
                        self.global_step, exc,
                    )

            if (step + 1) % self.config.gradient_accumulation_steps == 0:

                if self.scaler.is_enabled():
                    self.scaler.unscale_(self.optimizer)

                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.max_grad_norm,
                )

                if self.scaler.is_enabled():
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()

                if self.scheduler:
                    self.scheduler.step()

                self.optimizer.zero_grad(set_to_none=True)

            if (step + 1) % self.config.log_every_steps == 0:
                logger.info("step %d | loss %.6f", step + 1, float(raw_loss.detach().item()))

        # Final step fix — flush any partial accumulation window (M3)
        if step >= 0 and (step + 1) % self.config.gradient_accumulation_steps != 0:

            if self.scaler.is_enabled():
                self.scaler.unscale_(self.optimizer)

            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.config.max_grad_norm,
            )

            if self.scaler.is_enabled():
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                self.optimizer.step()

            if self.scheduler:
                self.scheduler.step()

            self.optimizer.zero_grad(set_to_none=True)

        mean_loss = (loss_accum / max(step_count, 1)).detach().item()
        return float(mean_loss)


    def _validate_epoch(self, dataloader: DataLoader) -> float:

        self.model.eval()

        loss_accum = torch.zeros((), device=self.device, dtype=torch.float32)
        step_count = 0

        with torch.no_grad(): 

            for batch in dataloader:

                batch = self._move_batch_to_device(batch)

                outputs = self.model(**self._prepare_model_inputs(batch))
                loss = self._extract_loss(outputs)

                loss_accum = loss_accum + loss.detach().to(loss_accum.dtype)
                step_count += 1

        mean_loss = (loss_accum / max(step_count, 1)).detach().item()
        return float(mean_loss)


    def _extract_loss(self, outputs):

        if isinstance(outputs, dict):
            if "loss" not in outputs:
                raise RuntimeError("Model output must contain 'loss'")
            return outputs["loss"]

        if hasattr(outputs, "loss"):
            return outputs.loss

        raise RuntimeError("Unable to extract loss")


    def _move_batch_to_device(self, batch):

        if isinstance(batch, dict):
            return move_to_device(batch, self.device)

        if isinstance(batch, (list, tuple)):
            return type(batch)(
                move_to_device(x, self.device) for x in batch
            )

        raise TypeError("Unsupported batch format")


    def _prepare_model_inputs(self, batch):

        if not isinstance(batch, dict):
            return batch

        if self._forward_accepts_kwargs:
            return batch

        forward_kwargs = {}

        for key, value in batch.items():

            if key in self._forward_params:
                forward_kwargs[key] = value

        return forward_kwargs