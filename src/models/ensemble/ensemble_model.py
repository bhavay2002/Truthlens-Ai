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
        # P2: move sub-models to the ensemble's device exactly once,
        # at construction time. The previous code did
        # ``model = model.to(self.device)`` inside ``forward``, which
        # both incurred a device-check on every call AND quietly
        # reassigned the local variable instead of materialising the
        # move on the registered ``nn.ModuleList`` entry.
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
            output = model(*args, **kwargs)
            logits = extract_logits(output)
            logits_list.append(logits)

        stacked = torch.stack(logits_list, dim=0)

        if self.config.strategy == "majority_vote":
            # `_majority_vote` returns per-class vote *counts*. These are
            # not real-valued logits, so feeding them into softmax (the
            # general-strategy code path below) would smear the
            # probability mass across all classes — exactly the bug the
            # audit flagged. Convert counts directly to a normalized
            # probability distribution and recover a logit-shaped tensor
            # by taking the log of those probabilities.
            vote_counts = self._majority_vote(stacked)
            denom = vote_counts.sum(dim=-1, keepdim=True).clamp(min=1.0)
            probs = vote_counts / denom
            logits = torch.log(probs.clamp(min=1e-12))

        elif self.config.strategy == "weighted" and self._weights is not None:
            weights = self._weights.to(stacked.device).view(
                -1, *([1] * (stacked.dim() - 1))
            )
            logits = (stacked * weights).sum(dim=0) / weights.sum()
            probs = F.softmax(logits, dim=-1)

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
        """Return per-class vote *counts* (not probabilities or logits).

        ``stacked`` is shape ``[num_models, ..., num_classes]``. For each
        model we take its ``argmax`` over the class dimension and add a
        one-hot vote into the accumulator. The caller is responsible for
        turning these counts into a probability distribution.
        """

        predictions = stacked.argmax(dim=-1)

        vote_counts = torch.zeros_like(stacked[0])

        for pred in predictions:
            one_hot = torch.zeros_like(vote_counts)
            one_hot.scatter_(-1, pred.unsqueeze(-1), 1.0)
            vote_counts += one_hot

        return vote_counts

    # =====================================================
    # UTILS
    # =====================================================

    def add_model(self, model: nn.Module) -> None:
        # P2: keep the device-placement invariant — every member of
        # ``self.models`` lives on ``self.device`` — true for any model
        # added after construction, not just those passed to ``__init__``.
        self.models.append(model.to(self.device))

    def get_num_models(self) -> int:
        return len(self.models)