"""
stacking_ensemble.py
Module: models.ensemble
Description:
    Stacking ensemble that uses a meta-learner on top of concatenated base-model
    outputs.
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
class StackingEnsembleConfig:
    """
    Configuration for StackingEnsembleModel.

    Attributes
    ----------
    device : str
        Target device.
    """

    device: str = "cpu"


class StackingEnsembleModel(nn.Module):
    """
    Combines base-model logits via a meta-learner (stacking).

    The concatenated logit outputs from all base models are fed into a
    meta-model that produces the final prediction.
    """

    def __init__(
        self,
        base_models: List[nn.Module],
        meta_model: nn.Module,
        config: Optional[StackingEnsembleConfig] = None,
    ) -> None:
        super().__init__()

        if not base_models:
            raise ValueError("At least one base model must be provided.")
        if meta_model is None:
            raise ValueError("A meta_model must be provided for stacking.")

        if config is None:
            config = StackingEnsembleConfig()

        self.config = config
        self.base_models = nn.ModuleList(base_models)
        self.meta_model = meta_model

        logger.info(
            "StackingEnsembleModel initialised | base_models=%d",
            len(base_models),
        )

    def forward(self, *args, **kwargs) -> torch.Tensor:
        """
        Run base models, concatenate their logits, pass through meta-model.

        Returns
        -------
        torch.Tensor
            Meta-model output logits.
        """
        all_logits: List[torch.Tensor] = [
            extract_logits(model(*args, **kwargs)) for model in self.base_models
        ]

        concatenated = torch.cat(all_logits, dim=-1)

        meta_output = self.meta_model(concatenated)
        return extract_logits(meta_output)
