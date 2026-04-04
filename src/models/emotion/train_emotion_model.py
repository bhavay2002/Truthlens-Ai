"""
Training helper for EmotionClassifier.
"""

from __future__ import annotations

import logging
from typing import Iterable, Dict, Any, Optional

import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.cuda.amp import autocast, GradScaler

logger = logging.getLogger(__name__)


class EmotionTrainer:
    """
    Lightweight trainer wrapper for EmotionClassifier.

    Supports:
    - device placement
    - gradient clipping
    - mixed precision (AMP)
    - scheduler updates
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: Optimizer,
        device: Optional[str] = None,
        scheduler: Optional[Any] = None,
        grad_clip: Optional[float] = None,
        use_amp: bool = False,
    ) -> None:

        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

        self.model = model.to(self.device)
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.grad_clip = grad_clip
        self.use_amp = use_amp

        self.scaler = GradScaler(enabled=use_amp)

        logger.info("EmotionTrainer initialized on device: %s", self.device)

    # -----------------------------------------------------

    def _move_batch(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Move tensors to correct device."""
        return {k: v.to(self.device) for k, v in batch.items()}

    # -----------------------------------------------------

    def train_step(self, batch: Dict[str, torch.Tensor]) -> float:

        self.model.train()

        batch = self._move_batch(batch)

        self.optimizer.zero_grad()

        with autocast(enabled=self.use_amp):

            outputs: Dict[str, Any] = self.model(**batch)

            loss = outputs.get("loss")

            if loss is None:
                raise RuntimeError("Model output did not include 'loss'")

        if self.use_amp:

            self.scaler.scale(loss).backward()

            if self.grad_clip is not None:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)

            self.scaler.step(self.optimizer)
            self.scaler.update()

        else:

            loss.backward()

            if self.grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)

            self.optimizer.step()

        if self.scheduler is not None:
            self.scheduler.step()

        return float(loss.item())

    # -----------------------------------------------------

    def fit_epoch(
        self,
        dataloader: Iterable[Dict[str, torch.Tensor]],
    ) -> float:
        """
        Train model for one epoch.
        """

        total_loss = 0.0
        steps = 0

        for batch in dataloader:

            loss = self.train_step(batch)

            total_loss += loss
            steps += 1

        if steps == 0:
            logger.warning("Empty dataloader received in fit_epoch")
            return 0.0

        avg_loss = total_loss / steps

        logger.debug("Epoch training loss: %.6f", avg_loss)

        return float(avg_loss)


__all__ = ["EmotionTrainer"]