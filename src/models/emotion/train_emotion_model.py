"""Simple training helper for EmotionClassifier."""

from __future__ import annotations

from typing import Iterable, Dict, Any

import torch
import torch.nn as nn
from torch.optim import Optimizer


class EmotionTrainer:
    """Tiny trainer wrapper used by legacy imports/tests."""

    def __init__(self, model: nn.Module, optimizer: Optimizer) -> None:
        self.model = model
        self.optimizer = optimizer

    def train_step(self, batch: Dict[str, torch.Tensor]) -> float:
        self.model.train()
        self.optimizer.zero_grad()

        outputs: Dict[str, Any] = self.model(**batch)
        loss = outputs.get("loss")
        if loss is None:
            raise RuntimeError("Model output did not include 'loss'")

        loss.backward()
        self.optimizer.step()
        return float(loss.item())

    def fit_epoch(self, dataloader: Iterable[Dict[str, torch.Tensor]]) -> float:
        losses = [self.train_step(batch) for batch in dataloader]
        if not losses:
            return 0.0
        return float(sum(losses) / len(losses))


__all__ = ["EmotionTrainer"]
