from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List, Optional, Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.ensemble._utils import extract_logits

logger = logging.getLogger(__name__)


# =========================================================
# CONFIG
# =========================================================

@dataclass
class EnsembleConfig:
    strategy: str = "average"  # average | weighted | majority_vote
    weights: Optional[List[float]] = None
    device: str = "cpu"
    return_probabilities: bool = True


# =========================================================
# MODEL
# =========================================================

class EnsembleModel(nn.Module):

    def __init__(
        self,
        models: List[nn.Module],
        config: Optional[EnsembleConfig] = None,
    ) -> None:
        super().__init__()

        if not models:
            raise ValueError("At least one model must be provided")

        self.config = config or EnsembleConfig()
        self.models = nn.ModuleList(models)

        if self.config.weights is not None:
            if len(self.config.weights) != len(models):
                raise ValueError("weights length mismatch")

            weights_tensor = torch.tensor(
                self.config.weights,
                dtype=torch.float32,
            )
            self.register_buffer("_weights", weights_tensor)
        else:
            self._weights = None

        self.device = torch.device(self.config.device)
        self.to(self.device)

        logger.info(
            "EnsembleModel | strategy=%s | models=%d",
            self.config.strategy,
            len(models),
        )

    # =====================================================
    # FORWARD
    # =====================================================

    def forward(self, *args, **kwargs) -> Dict[str, Any]:

        logits_list: List[torch.Tensor] = []

        for model in self.models:
            model = model.to(self.device)
            output = model(*args, **kwargs)
            logits = extract_logits(output)
            logits_list.append(logits)

        stacked = torch.stack(logits_list, dim=0)

        if self.config.strategy == "majority_vote":
            logits = self._majority_vote(stacked)

        elif self.config.strategy == "weighted" and self._weights is not None:
            weights = self._weights.to(stacked.device).view(
                -1, *([1] * (stacked.dim() - 1))
            )
            logits = (stacked * weights).sum(dim=0) / weights.sum()

        else:
            logits = stacked.mean(dim=0)

        probs = F.softmax(logits, dim=-1)
        preds = probs.argmax(dim=-1)
        confidence = probs.max(dim=-1).values
        entropy = -torch.sum(probs * torch.log(probs + 1e-12), dim=-1)

        return {
            "logits": logits,
            "probabilities": probs if self.config.return_probabilities else None,
            "predictions": preds,
            "confidence": confidence,
            "entropy": entropy,
        }

    # =====================================================
    # MAJORITY VOTE
    # =====================================================

    def _majority_vote(self, stacked: torch.Tensor) -> torch.Tensor:

        predictions = stacked.argmax(dim=-1)
        num_classes = stacked.size(-1)

        vote_logits = torch.zeros_like(stacked[0])

        for pred in predictions:
            one_hot = torch.zeros_like(vote_logits)
            one_hot.scatter_(-1, pred.unsqueeze(-1), 1.0)
            vote_logits += one_hot

        return vote_logits

    # =====================================================
    # UTILS
    # =====================================================

    def add_model(self, model: nn.Module) -> None:
        self.models.append(model)

    def get_num_models(self) -> int:
        return len(self.models)