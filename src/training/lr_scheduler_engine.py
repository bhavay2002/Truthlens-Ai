s#rc\training\lr_scheduler_engine.py

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional, Dict, Any

import torch

logger = logging.getLogger(__name__)


# =========================================================
# CONFIG
# =========================================================

@dataclass
class LRSchedulerConfig:
    name: str = "linear"  # linear | cosine | plateau | constant | etc.
    step_mode: str = "step"  # step | epoch | metric

    num_training_steps: Optional[int] = None
    num_warmup_steps: Optional[int] = None
    warmup_ratio: float = 0.0

    # Plateau scheduler params
    plateau_mode: str = "min"
    plateau_factor: float = 0.5
    plateau_patience: int = 2
    plateau_min_lr: float = 1e-7

    # Adaptive LR control
    enable_adaptive: bool = True
    spike_lr_scale: float = 0.5
    health_lr_scale: float = 0.7
    min_lr: float = 1e-7


# =========================================================
# ENGINE
# =========================================================

class LRSchedulerEngine:
    """
    Centralized Learning Rate Control System.

    Responsibilities:
    - create scheduler
    - manage step logic
    - react to training signals (spikes, health)
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        config: Optional[LRSchedulerConfig] = None,
    ) -> None:

        self.optimizer = optimizer
        self.config = config or LRSchedulerConfig()

        self.scheduler = self._build_scheduler()

        logger.info(
            "LRSchedulerEngine initialized | name=%s | step_mode=%s",
            self.config.name,
            self.config.step_mode,
        )

    # =====================================================
    # BUILD
    # =====================================================

    def _build_scheduler(self):

        name = self.config.name.lower()

        # -------------------------
        # PyTorch Plateau
        # -------------------------
        if name == "plateau":
            return torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode=self.config.plateau_mode,
                factor=self.config.plateau_factor,
                patience=self.config.plateau_patience,
                min_lr=self.config.plateau_min_lr,
            )

        # -------------------------
        # Transformers schedulers
        # -------------------------
        try:
            from transformers import (
                get_linear_schedule_with_warmup,
                get_cosine_schedule_with_warmup,
                get_cosine_with_hard_restarts_schedule_with_warmup,
                get_polynomial_decay_schedule_with_warmup,
                get_constant_schedule,
                get_constant_schedule_with_warmup,
            )
        except ImportError:
            raise ImportError("transformers required for scheduler")

        scheduler_map = {
            "linear": get_linear_schedule_with_warmup,
            "cosine": get_cosine_schedule_with_warmup,
            "cosine_restarts": get_cosine_with_hard_restarts_schedule_with_warmup,
            "polynomial": get_polynomial_decay_schedule_with_warmup,
            "constant": get_constant_schedule,
            "constant_with_warmup": get_constant_schedule_with_warmup,
        }

        if name not in scheduler_map:
            raise ValueError(f"Unsupported scheduler: {name}")

        fn = scheduler_map[name]

        # warmup calc
        warmup_steps = self.config.num_warmup_steps
        if warmup_steps is None:
            if self.config.num_training_steps and self.config.warmup_ratio > 0:
                warmup_steps = int(
                    self.config.num_training_steps * self.config.warmup_ratio
                )
            else:
                warmup_steps = 0

        if name == "constant":
            return fn(self.optimizer)

        if name == "constant_with_warmup":
            return fn(self.optimizer, num_warmup_steps=warmup_steps)

        if self.config.num_training_steps is None:
            raise ValueError(f"{name} requires num_training_steps")

        return fn(
            self.optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=self.config.num_training_steps,
        )

    # =====================================================
    # STEP
    # =====================================================

    def step(
        self,
        *,
        metric: Optional[float] = None,
    ) -> None:

        if self.scheduler is None:
            return

        if self.config.step_mode == "step":
            self.scheduler.step()

        elif self.config.step_mode == "epoch":
            self.scheduler.step()

        elif self.config.step_mode == "metric":
            if metric is None:
                raise ValueError("metric required for plateau scheduler")
            self.scheduler.step(metric)

        else:
            raise ValueError(f"Invalid step_mode: {self.config.step_mode}")

    # =====================================================
    # ADAPTIVE CONTROL (KEY FEATURE)
    # =====================================================

    def adapt(
        self,
        monitoring_metrics: Dict[str, Any],
    ) -> None:
        """
        Adjust LR based on training signals.
        """

        if not self.config.enable_adaptive:
            return

        current_lr = self.get_lr()

        # -------------------------
        # Spike handling
        # -------------------------
        if monitoring_metrics.get("spike"):
            new_lr = max(
                current_lr * self.config.spike_lr_scale,
                self.config.min_lr,
            )
            self._set_lr(new_lr)
            logger.warning(f"LR reduced due to spike: {current_lr} -> {new_lr}")
            return

        # -------------------------
        # Health degradation
        # -------------------------
        health = monitoring_metrics.get("health")

        if health is not None and health < 0.3:
            new_lr = max(
                current_lr * self.config.health_lr_scale,
                self.config.min_lr,
            )
            self._set_lr(new_lr)
            logger.warning(f"LR reduced due to low health: {current_lr} -> {new_lr}")

    # =====================================================
    # LR UTILITIES
    # =====================================================

    def get_lr(self) -> float:
        if not self.optimizer.param_groups:
            return 0.0
        return self.optimizer.param_groups[0]["lr"]

    def _set_lr(self, lr: float) -> None:
        for group in self.optimizer.param_groups:
            group["lr"] = lr

    # =====================================================
    # STATE
    # =====================================================

    def state_dict(self) -> Dict[str, Any]:
        if self.scheduler:
            return self.scheduler.state_dict()
        return {}

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        if self.scheduler:
            self.scheduler.load_state_dict(state_dict)