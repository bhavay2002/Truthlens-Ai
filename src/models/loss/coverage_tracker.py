from __future__ import annotations

import logging
from typing import Dict

import torch

logger = logging.getLogger(__name__)


class EMACoverageTracker:
    """
    EMA-based task coverage tracker for multi-task imbalance correction.
    """

    def __init__(
        self,
        alpha: float = 0.1,
        floor: float = 0.05,
        cap: float = 10.0,
        enabled: bool = True,
        warmup_steps: int = 0,  # ✅ optional (no behavior change by default)
    ) -> None:

        if not 0.0 < alpha <= 1.0:
            raise ValueError(f"alpha must be in (0,1], got {alpha}")

        if floor <= 0.0:
            raise ValueError(f"floor must be positive, got {floor}")

        if cap < 1.0:
            raise ValueError(f"cap must be >= 1.0, got {cap}")

        if warmup_steps < 0:
            raise ValueError(f"warmup_steps must be >= 0, got {warmup_steps}")

        self.alpha = float(alpha)
        self.floor = float(floor)
        self.cap = float(cap)
        self.enabled = enabled
        self.warmup_steps = int(warmup_steps)

        # per-task EMA coverage
        self._coverage: Dict[str, float] = {}

        # diagnostics
        self._steps: Dict[str, int] = {}

        logger.info(
            "EMACoverageTracker initialized | alpha=%.3f floor=%.3f cap=%.2f enabled=%s warmup=%d",
            self.alpha,
            self.floor,
            self.cap,
            self.enabled,
            self.warmup_steps,
        )

    # =========================================================
    # UPDATE COVERAGE
    # =========================================================

    def update(
        self,
        task: str,
        labels: torch.Tensor,
        ignore_index: float | int = -100,
        task_type: str = "multi_class",
    ) -> None:

        if not self.enabled:
            return

        if not torch.is_tensor(labels) or labels.numel() == 0:
            has_label = False

        else:
            if task_type == "multi_class":
                has_label = bool(labels.ne(ignore_index).any())

            elif task_type in {"binary", "multi_label"}:
                has_label = bool(labels.ne(float(ignore_index)).any())

            elif task_type == "regression":
                has_label = True

            else:
                raise ValueError(f"Unknown task_type: {task_type}")

        prev = self._coverage.get(task, 0.0)

        new_cov = (
            self.alpha * (1.0 if has_label else 0.0)
            + (1.0 - self.alpha) * prev
        )

        self._coverage[task] = new_cov
        self._steps[task] = self._steps.get(task, 0) + 1

    # =========================================================
    # APPLY WEIGHTING
    # =========================================================

    def weight(
        self,
        task: str,
        loss: torch.Tensor,
    ) -> torch.Tensor:

        if not self.enabled:
            return loss

        if not torch.is_tensor(loss):
            raise TypeError("loss must be tensor")

        if not torch.isfinite(loss):
            raise RuntimeError(f"Non-finite loss detected for task '{task}'")

        step = self._steps.get(task, 0)

        # ---- warmup (no aggressive boosting early) ----
        if step < self.warmup_steps:
            return loss

        cov = max(self._coverage.get(task, 0.0), self.floor)
        multiplier = min(1.0 / cov, self.cap)

        # ---- device-safe tensor ----
        multiplier_tensor = loss.new_tensor(multiplier)

        weighted = loss * multiplier_tensor

        return weighted

    # =========================================================
    # UTILITIES
    # =========================================================

    def get_coverage(self) -> Dict[str, float]:
        return dict(self._coverage)

    def get_multipliers(self) -> Dict[str, float]:
        result = {}

        for t, cov in self._coverage.items():
            cov_safe = max(cov, self.floor)
            result[t] = min(1.0 / cov_safe, self.cap)

        return result

    def reset(self) -> None:
        self._coverage.clear()
        self._steps.clear()

        logger.info("EMACoverageTracker state reset")