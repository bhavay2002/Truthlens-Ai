"""
ensemble_model.py
Module: models.ensemble
Description:
    Provides average-based and majority-vote ensemble strategies for combining
    the outputs of multiple classification models in the TruthLens AI system.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import List, Optional

import torch
import torch.nn as nn

from src.models.ensemble._utils import extract_logits

logger = logging.getLogger(__name__)


@dataclass
class EnsembleConfig:
    """
    Configuration for EnsembleModel.

    Attributes
    ----------
    strategy : str
        Combination strategy: 'average' or 'majority_vote'.
    weights : Optional[List[float]]
        Per-model weights for weighted-average variant (ignored for majority_vote).
    device : str
        Device to run inference on.
    """

    strategy: str = "average"
    weights: Optional[List[float]] = None
    device: str = "cpu"


class EnsembleModel(nn.Module):
    """
    Combines predictions from multiple models using a configurable strategy.

    Supported strategies
    --------------------
    'average'       : Averages logits across all models.
    'majority_vote' : Uses argmax of summed one-hot votes.
    """

    def __init__(
        self,
        models: List[nn.Module],
        config: Optional[EnsembleConfig] = None,
    ) -> None:
        super().__init__()

        if not models:
            raise ValueError("At least one model must be provided.")

        if config is None:
            config = EnsembleConfig()

        self.config = config
        self.models = nn.ModuleList(models)

        if config.weights is not None:
            if len(config.weights) != len(models):
                raise ValueError(
                    "Length of weights must match the number of models."
                )
            weights_tensor = torch.tensor(
                config.weights, dtype=torch.float32
            )
            self.register_buffer("_weights", weights_tensor)
        else:
            self._weights: Optional[torch.Tensor] = None

        logger.info(
            "EnsembleModel initialised | strategy=%s models=%d",
            config.strategy,
            len(models),
        )

    def forward(self, *args, **kwargs) -> torch.Tensor:
        """
        Run all base models and aggregate their logits.

        Returns
        -------
        torch.Tensor
            Aggregated logits tensor.
        """
        all_logits: List[torch.Tensor] = [
            extract_logits(model(*args, **kwargs)) for model in self.models
        ]

        stacked = torch.stack(all_logits, dim=0)

        if self.config.strategy == "majority_vote":
            return self._majority_vote(stacked)

        if self._weights is not None:
            weights = self._weights.to(stacked.device).view(
                -1, *([1] * (stacked.dim() - 1))
            )
            return (stacked * weights).sum(dim=0) / weights.sum()

        return stacked.mean(dim=0)

    def _majority_vote(self, stacked: torch.Tensor) -> torch.Tensor:
        """
        Aggregate via majority vote (returns summed one-hot logits).
        """
        predictions = stacked.argmax(dim=-1)
        num_classes = stacked.size(-1)
        vote_logits = torch.zeros_like(stacked[0])

        for pred in predictions:
            one_hot = torch.zeros_like(vote_logits)
            one_hot.scatter_(-1, pred.unsqueeze(-1), 1.0)
            vote_logits += one_hot

        return vote_logits
