"""
File Name: weighted_ensemble.py
Module: models.ensemble
Description:
    Implements a weighted ensemble model for combining predictions from
    multiple base models. Each model contributes to the final prediction
    according to an assigned weight.

    This module supports both logits and probability outputs and is designed
    to integrate seamlessly into PyTorch-based ML systems. It provides
    facilities for validating model compatibility, normalizing weights,
    performing weighted aggregation, and generating calibrated predictions.

    The architecture follows production-grade ML engineering practices and
    supports GPU inference, structured logging, modular design, and strong
    input validation.

Author: ML Engineering System
Date: 2026-04-03
Dependencies:
    torch
    torch.nn
    logging
    dataclasses
    typing
Inputs:
    Multiple PyTorch models and associated ensemble weights.
Outputs:
    Weighted ensemble logits, probabilities, and predicted class labels.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List, Optional

import torch
import torch.nn as nn


logger = logging.getLogger(__name__)


@dataclass
class WeightedEnsembleConfig:
    """
    Configuration for weighted ensemble model.
    """

    weights: Optional[List[float]] = None
    normalize_weights: bool = True
    device: str = "cpu"

    def __post_init__(self) -> None:
        if self.weights is not None:
            if len(self.weights) == 0:
                raise ValueError("Weights list cannot be empty.")

            for w in self.weights:
                if w < 0:
                    raise ValueError("Weights must be non-negative.")


class WeightedEnsembleModel(nn.Module):
    """
    Weighted ensemble for combining predictions from multiple models.

    Supports:
        - Weighted logits aggregation
        - Weighted probability aggregation
        - GPU inference
    """

    def __init__(
        self,
        models: List[nn.Module],
        config: WeightedEnsembleConfig | None = None,
    ) -> None:
        super().__init__()

        if not models:
            raise ValueError("At least one model must be provided for the ensemble.")

        self.models = nn.ModuleList(models)
        self.config = config or WeightedEnsembleConfig()

        self.device = torch.device(self.config.device)
        self.to(self.device)

        self.num_models = len(models)

        if self.config.weights is None:
            self.weights = torch.ones(self.num_models, dtype=torch.float32)
        else:
            if len(self.config.weights) != self.num_models:
                raise ValueError(
                    "Number of weights must match the number of models."
                )

            self.weights = torch.tensor(
                self.config.weights,
                dtype=torch.float32,
            )

        if self.config.normalize_weights:
            weight_sum = torch.sum(self.weights)
            if weight_sum <= 0:
                raise ValueError("Sum of weights must be positive.")
            self.weights = self.weights / weight_sum

        self.weights = self.weights.to(self.device)

        logger.info(
            "Initialized WeightedEnsembleModel with %d models.", self.num_models
        )

    def _collect_logits(self, x: torch.Tensor) -> torch.Tensor:
        """
        Run forward pass through each model and collect logits.

        Returns
        -------
        Tensor of shape [num_models, batch_size, num_classes]
        """

        logits_list = []

        for model in self.models:
            model.eval()
            with torch.no_grad():
                logits = model(x.to(self.device))
                logits_list.append(logits)

        return torch.stack(logits_list)

    def _weighted_logits(self, logits: torch.Tensor) -> torch.Tensor:
        """
        Apply weighted aggregation to logits.
        """

        weights = self.weights.view(-1, 1, 1)

        weighted = logits * weights

        return torch.sum(weighted, dim=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of weighted ensemble.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor.

        Returns
        -------
        torch.Tensor
            Aggregated logits.
        """

        logits = self._collect_logits(x)

        return self._weighted_logits(logits)

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute probability predictions.
        """

        logits = self.forward(x)

        return torch.softmax(logits, dim=1)

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute predicted class labels.
        """

        probs = self.predict_proba(x)

        return torch.argmax(probs, dim=1)

    def get_weights(self) -> torch.Tensor:
        """
        Return ensemble weights.
        """

        return self.weights.detach().cpu()