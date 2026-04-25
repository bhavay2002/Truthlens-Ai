from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, Any, Optional

import torch

logger = logging.getLogger(__name__)


# =========================================================
# CONFIG
# =========================================================

@dataclass
class MonitoringConfig:
    spike_threshold: float = 3.0
    ema_alpha: float = 0.1
    health_threshold: float = 0.3
    enable_grad_monitor: bool = False
    grad_monitor_interval: int = 200


# =========================================================
# HELPERS
# =========================================================

class LossEMA:
    def __init__(self, alpha: float):
        self.alpha = alpha
        self.value: Optional[float] = None

    def update(self, x: float) -> float:
        if self.value is None:
            self.value = x
        else:
            self.value = self.alpha * x + (1 - self.alpha) * self.value
        return self.value


class SpikeDetector:
    def __init__(self, threshold: float):
        self.threshold = threshold

    def is_spike(self, loss: float, ema: float) -> bool:
        if ema is None or ema == 0:
            return False
        ratio = loss / (ema + 1e-12)
        return ratio > self.threshold


class HealthScore:
    def compute(
        self,
        loss: float,
        ema: float,
        grad_norm: Optional[float] = None,
    ) -> float:

        if ema is None:
            return 1.0

        stability = 1.0 - min(abs(loss - ema) / (ema + 1e-9), 1.0)

        grad_penalty = 0.0
        if grad_norm is not None:
            if grad_norm > 10:
                grad_penalty = min((grad_norm - 10) / 50, 1.0)

        return max(0.0, stability - grad_penalty)


# =========================================================
# MONITORING ENGINE
# =========================================================

class MonitoringEngine:
    """
    Training observability engine.

    Responsibilities:
    - track loss (EMA)
    - detect spikes
    - compute health score
    - optional gradient monitoring
    """

    def __init__(self, config: Optional[MonitoringConfig] = None):

        self.config = config or MonitoringConfig()

        self.loss_ema = LossEMA(self.config.ema_alpha)
        self.spike_detector = SpikeDetector(self.config.spike_threshold)
        self.health = HealthScore()

        self.step = 0

        logger.info("MonitoringEngine initialized")

    # =====================================================
    # UPDATE
    # =====================================================

    def update(
        self,
        outputs: Dict[str, Any],
        model: Optional[torch.nn.Module] = None,
    ) -> Dict[str, Any]:

        self.step += 1

        loss = self._extract_loss(outputs)

        ema = self.loss_ema.update(loss)

        spike = self.spike_detector.is_spike(loss, ema)

        grad_norm = None

        if (
            self.config.enable_grad_monitor
            and model is not None
            and self.step % self.config.grad_monitor_interval == 0
        ):
            grad_norm = self._compute_grad_norm(model)

        health_score = self.health.compute(loss, ema, grad_norm)

        # -------------------------
        # ACTIONABLE RESPONSE
        # -------------------------

        action = None

        if health_score < self.config.health_threshold:
            action = "reduce_lr"

        if spike:
            action = "spike_detected"

        metrics = {
            "loss": loss,
            "ema_loss": ema,
            "spike": spike,
            "grad_norm": grad_norm,
            "health": health_score,
            "action": action,
        }

        return metrics

    # =====================================================
    # LOSS
    # =====================================================

    def _extract_loss(self, outputs: Dict[str, Any]) -> float:

        loss = outputs.get("raw_loss") or outputs.get("loss")

        if isinstance(loss, torch.Tensor):
            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite loss detected: {loss.item()}")
            return float(loss.detach().item())

        if isinstance(loss, (int, float)):
            return float(loss)

        raise RuntimeError("Loss not found in outputs")

    # =====================================================
    # GRADIENTS
    # =====================================================

    def _compute_grad_norm(self, model: torch.nn.Module) -> float:

        total_norm = 0.0

        for p in model.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2

        return total_norm ** 0.5