from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


# =========================================================
# CONFIG
# =========================================================

@dataclass
class CheckpointConfig:
    directory: str
    max_checkpoints: int = 5
    save_optimizer: bool = True
    save_scheduler: bool = True
    monitor_metric: Optional[str] = None
    mode: str = "min"  # "min" | "max"


# =========================================================
# CHECKPOINT ENGINE
# =========================================================

class CheckpointEngine:
    """
    Modular checkpoint engine.

    Responsibilities:
    - save checkpoints
    - load checkpoints
    - track best model
    - cleanup old checkpoints
    """

    def __init__(self, config: CheckpointConfig):

        self.config = config

        self.dir = Path(config.directory)
        self.dir.mkdir(parents=True, exist_ok=True)

        self.best_metric: Optional[float] = None
        self.best_path: Optional[Path] = None

        logger.info("CheckpointEngine initialized | dir=%s", self.dir)

    # =====================================================
    # SAVE
    # =====================================================

    def save(
        self,
        step: int,
        model: nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[Any] = None,
        metrics: Optional[Dict[str, float]] = None,
        tag: Optional[str] = None,
    ) -> Path:

        filename = f"checkpoint_step_{step}.pt"
        if tag:
            filename = f"{tag}_{filename}"

        path = self.dir / filename

        state = {
            "step": step,
            "model_state_dict": model.state_dict(),
            "metrics": metrics or {},
        }

        if self.config.save_optimizer and optimizer is not None:
            state["optimizer_state_dict"] = optimizer.state_dict()

        if self.config.save_scheduler and scheduler is not None:
            state["scheduler_state_dict"] = scheduler.state_dict()

        torch.save(state, path)

        logger.info("Checkpoint saved: %s", path)

        # Track best model
        self._update_best(path, metrics)

        # Cleanup
        self._cleanup()

        return path

    # =====================================================
    # LOAD
    # =====================================================

    def load(
        self,
        path: str | Path,
        model: nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[Any] = None,
        map_location: Optional[str] = None,
    ) -> Dict[str, Any]:

        path = Path(path)

        if not path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {path}")

        checkpoint = torch.load(
            path,
            map_location=map_location or "cpu",
            weights_only=False,
        )

        model.load_state_dict(checkpoint["model_state_dict"])

        if optimizer and "optimizer_state_dict" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        if scheduler and "scheduler_state_dict" in checkpoint:
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        logger.info("Checkpoint loaded: %s", path)

        return checkpoint

    # =====================================================
    # BEST MODEL
    # =====================================================

    def _update_best(
        self,
        path: Path,
        metrics: Optional[Dict[str, float]],
    ):

        if not self.config.monitor_metric or not metrics:
            return

        metric_name = self.config.monitor_metric

        if metric_name not in metrics:
            return

        value = metrics[metric_name]

        if self.best_metric is None:
            is_better = True
        elif self.config.mode == "min":
            is_better = value < self.best_metric
        else:
            is_better = value > self.best_metric

        if is_better:
            self.best_metric = value
            self.best_path = path

            best_path = self.dir / "best_model.pt"
            torch.save(torch.load(path), best_path)

            logger.info(
                "New best model | %s=%.6f saved to %s",
                metric_name,
                value,
                best_path,
            )

    # =====================================================
    # CLEANUP
    # =====================================================

    def _cleanup(self):

        checkpoints = sorted(
            self.dir.glob("checkpoint_step_*.pt"),
            key=lambda p: p.stat().st_mtime,
        )

        if len(checkpoints) <= self.config.max_checkpoints:
            return

        to_delete = checkpoints[:-self.config.max_checkpoints]

        for path in to_delete:
            try:
                path.unlink()
                logger.debug("Deleted old checkpoint: %s", path)
            except Exception as e:
                logger.warning("Failed to delete checkpoint %s: %s", path, e)

    # =====================================================
    # UTIL
    # =====================================================

    def latest_checkpoint(self) -> Optional[Path]:

        checkpoints = sorted(
            self.dir.glob("checkpoint_step_*.pt"),
            key=lambda p: p.stat().st_mtime,
        )

        if not checkpoints:
            return None

        return checkpoints[-1]