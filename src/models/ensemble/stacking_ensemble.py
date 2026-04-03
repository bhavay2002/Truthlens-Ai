"""
File Name: stacking_ensemble.py
Module: models.ensemble
Description:
    Implements a stacking ensemble architecture for combining predictions
    from multiple base models using a meta-learner. In stacking, base models
    first produce intermediate predictions (logits or probabilities), which
    are then used as features for a second-level model (meta-model) that
    produces the final prediction.

    This implementation supports PyTorch-based base models and a PyTorch
    meta-model, enabling end-to-end GPU execution. It includes structured
    logging, input validation, and modular design suitable for research and
    production ML systems.

Author: ML Engineering System
Date: 2026-04-03
Dependencies:
    torch
    torch.nn
    logging
    dataclasses
    typing
Inputs:
    Base model outputs (logits or probabilities) and input features.
Outputs:
    Final stacked predictions from the meta-model.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List, Optional

import torch
import torch.nn as nn


logger = logging.getLogger(__name__)


@dataclass
class StackingEnsembleConfig:
    """
    Configuration for stacking ensemble behavior.
    """

    use_probabilities: bool = True
    device: str = "cpu"

    def __post_init__(self) -> None:
        if not isinstance(self.use_probabilities, bool):
            raise ValueError("use_probabilities must be a boolean value.")


class StackingEnsembleModel(nn.Module):
    """
    Stacking ensemble model combining base model predictions
    through a meta-learner.
    """

    def __init__(
        self,
        base_models: List[nn.Module],
        meta_model: nn.Module,
        config: Optional[StackingEnsembleConfig] = None,
    ) -> None:
        super().__init__()

        if not base_models:
            raise ValueError("At least one base model is required.")

        self.base_models = nn.ModuleList(base_models)
        self.meta_model = meta_model
        self.config = config or StackingEnsembleConfig()

        self.device = torch.device(self.config.device)
        self.to(self.device)

        logger.info(
            "Initialized StackingEnsembleModel with %d base models.",
            len(self.base_models),
        )

    def _collect_base_outputs(self, x: torch.Tensor) -> torch.Tensor:
        """
        Run forward pass through base models and collect outputs.

        Returns
        -------
        torch.Tensor
            Shape: [batch_size, num_models * num_classes]
        """

        outputs = []

        for model in self.base_models:
            model.eval()

            with torch.no_grad():
                logits = model(x.to(self.device))

                if self.config.use_probabilities:
                    preds = torch.softmax(logits, dim=1)
                else:
                    preds = logits

                outputs.append(preds)

        stacked = torch.cat(outputs, dim=1)

        return stacked

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of stacking ensemble.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor.

        Returns
        -------
        torch.Tensor
            Meta-model output logits.
        """

        base_features = self._collect_base_outputs(x)

        meta_logits = self.meta_model(base_features)

        return meta_logits

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """
        Return probability predictions from the stacked model.
        """

        logits = self.forward(x)

        return torch.softmax(logits, dim=1)

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """
        Return predicted class labels.
        """

        probs = self.predict_proba(x)

        return torch.argmax(probs, dim=1)

    def get_base_models(self) -> List[nn.Module]:
        """
        Return base models used in the ensemble.
        """

        return list(self.base_models)

    def get_meta_model(self) -> nn.Module:
        """
        Return the meta-learner model.
        """

        return self.meta_model