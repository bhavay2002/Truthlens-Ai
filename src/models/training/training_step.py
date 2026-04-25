from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional, Any

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


@dataclass
class TrainingStepConfig:
    gradient_accumulation_steps: int = 1
    max_grad_norm: float = 1.0
    use_mixed_precision: bool = True


class TrainingStep:
    """
    Pure optimization engine.

    Responsibilities:
    - backward pass
    - gradient scaling (AMP)
    - gradient clipping
    - optimizer step
    - scheduler step

    Does NOT:
    - forward pass
    - loss computation
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[Any],
        config: TrainingStepConfig,
    ) -> None:

        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.config = config

        self.use_amp = config.use_mixed_precision and torch.cuda.is_available()

        self.scaler = torch.cuda.amp.GradScaler(
            enabled=self.use_amp
        )

        logger.info("TrainingStep initialized")

    # =====================================================
    # MAIN STEP
    # =====================================================

    def backward_and_step(
        self,
        loss: torch.Tensor,
        step: int,
    ) -> torch.Tensor:

        if not torch.isfinite(loss):
            raise RuntimeError(f"Invalid loss: {loss.item()}")

        scaled_loss = loss / self.config.gradient_accumulation_steps

        # -------------------------
        # BACKWARD
        # -------------------------

        if self.scaler.is_enabled():
            self.scaler.scale(scaled_loss).backward()
        else:
            scaled_loss.backward()

        # -------------------------
        # OPTIMIZER STEP
        # -------------------------

        should_step = (
            (step + 1) % self.config.gradient_accumulation_steps == 0
        )

        if should_step:

            if self.scaler.is_enabled():
                self.scaler.unscale_(self.optimizer)

            if self.config.max_grad_norm is not None:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.max_grad_norm,
                )

            if self.scaler.is_enabled():
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                self.optimizer.step()

            if self.scheduler is not None:
                try:
                    self.scheduler.step()
                except TypeError:
                    self.scheduler.step(float(loss.detach().item()))

            self.optimizer.zero_grad(set_to_none=True)

        return loss.detach()