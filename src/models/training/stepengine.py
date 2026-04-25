from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, Optional

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


# =========================================================
# CONFIG
# =========================================================

@dataclass
class StepEngineConfig:
    gradient_accumulation_steps: int = 1
    max_grad_norm: Optional[float] = 1.0
    use_amp: bool = True
    amp_dtype: str = "bf16"  # "fp16" | "bf16"
    scheduler_step_mode: str = "step"  # "step" | "epoch" | "metric"


# =========================================================
# STEP ENGINE
# =========================================================

class StepEngine:
    """
    Modular training step engine.

    Responsibilities:
    - forward pass
    - loss scaling (grad accumulation)
    - backward pass
    - optimizer step
    - AMP handling
    - gradient clipping
    - scheduler stepping

    Does NOT:
    - compute metrics
    - logging
    - checkpointing
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[Any] = None,
        config: Optional[StepEngineConfig] = None,
        loss_engine: Optional[Any] = None,
        device: Optional[str] = None,
    ) -> None:

        if not isinstance(model, nn.Module):
            raise TypeError("model must be nn.Module")

        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.config = config or StepEngineConfig()
        self.loss_engine = loss_engine

        self.device = torch.device(
            device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        )

        self.model.to(self.device)

        # AMP setup
        self.use_amp = self.config.use_amp and self.device.type == "cuda"

        if self.config.amp_dtype == "bf16":
            self.autocast_dtype = torch.bfloat16
        else:
            self.autocast_dtype = torch.float16

        self.scaler = torch.cuda.amp.GradScaler(
            enabled=self.use_amp and self.autocast_dtype == torch.float16
        )

        logger.info(
            "StepEngine initialized | device=%s | amp=%s",
            self.device,
            self.use_amp,
        )

    # =====================================================
    # STEP
    # =====================================================

    def step(
        self,
        batch: Dict[str, torch.Tensor],
        step_idx: int,
    ) -> Dict[str, Any]:

        self.model.train()

        batch = self._move_batch(batch)

        # -------------------------
        # FORWARD
        # -------------------------

        with torch.autocast(
            device_type=self.device.type,
            dtype=self.autocast_dtype,
            enabled=self.use_amp,
        ):
            outputs = self.model(**batch)

            raw_loss = self._compute_loss(outputs, batch)

            loss = raw_loss / self.config.gradient_accumulation_steps

        # -------------------------
        # BACKWARD
        # -------------------------

        if self.scaler.is_enabled():
            self.scaler.scale(loss).backward()
        else:
            loss.backward()

        # -------------------------
        # OPTIMIZER STEP
        # -------------------------

        should_step = (
            (step_idx + 1) % self.config.gradient_accumulation_steps == 0
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

            self.optimizer.zero_grad(set_to_none=True)

            self._step_scheduler(raw_loss)

        # -------------------------
        # OUTPUT
        # -------------------------

        return {
            "outputs": outputs,
            "loss": loss.detach(),
            "raw_loss": raw_loss.detach(),
        }

    # =====================================================
    # LOSS
    # =====================================================

    def _compute_loss(
        self,
        outputs: Dict[str, Any],
        batch: Dict[str, Any],
    ) -> torch.Tensor:

        # Prefer external loss engine (modular design)
        if self.loss_engine is not None:
            return self.loss_engine.compute(outputs, batch)

        # Fallback to model-provided loss
        loss = outputs.get("loss")

        if loss is None:
            raise RuntimeError("No loss found in outputs and no loss_engine provided")

        if not torch.isfinite(loss):
            raise RuntimeError(f"Non-finite loss detected: {loss.item()}")

        return loss

    # =====================================================
    # SCHEDULER
    # =====================================================

    def _step_scheduler(self, raw_loss: torch.Tensor) -> None:

        if self.scheduler is None:
            return

        mode = self.config.scheduler_step_mode

        if mode == "step":
            self.scheduler.step()

        elif mode == "metric":
            self.scheduler.step(float(raw_loss.item()))

        elif mode == "epoch":
            pass  # handled externally

        else:
            raise ValueError(f"Invalid scheduler mode: {mode}")

    # =====================================================
    # DEVICE
    # =====================================================

    def _move_batch(
        self,
        batch: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:

        return {
            k: v.to(self.device, non_blocking=True)
            if isinstance(v, torch.Tensor)
            else v
            for k, v in batch.items()
        }