"""
weighted_ensemble.py
Module: models.ensemble
Description:
    Weighted-average ensemble that assigns explicit per-model weights to logit
    combination.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List, Optional

import torch
import torch.nn as nn

from src.models.ensemble._utils import extract_logits

logger = logging.getLogger(__name__)


@dataclass
class WeightedEnsembleConfig:
    """
    Configuration for WeightedEnsembleModel.

    Attributes
    ----------
    weights : Optional[List[float]]
        Explicit per-model weights.  If None, equal weights are used.
    device : str
        Target device.
    """

    weights: Optional[List[float]] = None
    device: str = "cpu"


class WeightedEnsembleModel(nn.Module):
    """
    Combines model logits using explicit per-model weights.
    """

    def __init__(
        self,
        models: List[nn.Module],
        config: Optional[WeightedEnsembleConfig] = None,
    ) -> None:
        super().__init__()

        if not models:
            raise ValueError("At least one model must be provided.")

        if config is None:
            config = WeightedEnsembleConfig()

        self.config = config
        self.models = nn.ModuleList(models)

        if config.weights is not None:
            if len(config.weights) != len(models):
                raise ValueError(
                    "Length of weights must match the number of models."
                )
            weights_tensor = torch.tensor(config.weights, dtype=torch.float32)
        else:
            weights_tensor = torch.ones(len(models), dtype=torch.float32)

        self.register_buffer("_weights", weights_tensor)

        logger.info(
            "WeightedEnsembleModel initialised | models=%d",
            len(models),
        )

    def forward(self, *args, **kwargs) -> torch.Tensor:
        """
        Combine logits using per-model weights.

        Returns
        -------
        torch.Tensor
            Weighted-average logits.
        """
        all_logits: List[torch.Tensor] = [
            extract_logits(model(*args, **kwargs)) for model in self.models
        ]

        stacked = torch.stack(all_logits, dim=0)
        weights = self._weights.to(stacked.device).view(
            -1, *([1] * (stacked.dim() - 1))
        )

        return (stacked * weights).sum(dim=0) / weights.sum()
