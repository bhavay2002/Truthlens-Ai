"""
File Name: ensemble_model.py
Module: models.ensemble
Description:
    Implements a flexible ensemble model architecture for combining predictions
    from multiple machine learning models. The ensemble supports common
    strategies such as averaging, weighted averaging, and majority voting.

    Designed for research and production ML pipelines, this module integrates
    with PyTorch models and supports GPU inference, structured logging,
    configurable ensemble strategies, and reproducible evaluation.

Dependencies:
    torch
    torch.nn
    logging
    dataclasses
    typing
Inputs:
    Multiple model outputs (logits or probabilities).
Outputs:
    Ensemble predictions and probabilities.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List, Optional

import torch
import torch.nn as nn


logger = logging.getLogger(__name__)


@dataclass
class EnsembleConfig:
    """
    Configuration for ensemble behavior.
    """

    strategy: str = "average"
    weights: Optional[List[float]] = None
    device: str = "cpu"

    def __post_init__(self) -> None:
        valid_strategies = {"average", "weighted_average", "majority_vote"}

        if self.strategy not in valid_strategies:
            raise ValueError(
                f"Invalid ensemble strategy '{self.strategy}'. "
                f"Must be one of {valid_strategies}"
            )

        if self.strategy == "weighted_average" and self.weights is None:
            raise ValueError("Weights must be provided for weighted_average strategy.")


class EnsembleModel(nn.Module):
    """
    Ensemble wrapper for combining predictions from multiple models.
    """

    def __init__(
        self,
        models: List[nn.Module],
        config: EnsembleConfig | None = None,
    ) -> None:
        super().__init__()

        if not models:
            raise ValueError("Ensemble must contain at least one model.")

        self.models = nn.ModuleList(models)
        self.config = config or EnsembleConfig()
        self.device = torch.device(self.config.device)

        self.to(self.device)

        if self.config.weights is not None:
            if len(self.config.weights) != len(models):
                raise ValueError(
                    "Number of weights must match number of models in ensemble."
                )

            total = sum(self.config.weights)
            if total <= 0:
                raise ValueError("Sum of weights must be positive.")

            self.weights = torch.tensor(
                [w / total for w in self.config.weights],
                dtype=torch.float32,
                device=self.device,
            )
        else:
            self.weights = None

        logger.info(
            "Initialized EnsembleModel with %d models using strategy '%s'",
            len(models),
            self.config.strategy,
        )

    def _collect_logits(self, x: torch.Tensor) -> torch.Tensor:
        """
        Run forward pass through all models and collect logits.

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

    def _average(self, logits: torch.Tensor) -> torch.Tensor:
        """
        Average ensemble predictions.
        """

        return torch.mean(logits, dim=0)

    def _weighted_average(self, logits: torch.Tensor) -> torch.Tensor:
        """
        Weighted average ensemble predictions.
        """

        if self.weights is None:
            raise RuntimeError("Weights are not initialized for weighted averaging.")

        weights = self.weights.view(-1, 1, 1)

        weighted_logits = logits * weights

        return torch.sum(weighted_logits, dim=0)

    def _majority_vote(self, logits: torch.Tensor) -> torch.Tensor:
        """
        Majority voting ensemble.
        """

        predictions = torch.argmax(logits, dim=2)

        batch_size = predictions.shape[1]
        num_classes = logits.shape[2]

        votes = torch.zeros(
            batch_size,
            num_classes,
            device=self.device,
        )

        for model_preds in predictions:
            votes.scatter_add_(
                1,
                model_preds.unsqueeze(1),
                torch.ones_like(model_preds.unsqueeze(1), dtype=torch.float32),
            )

        return votes

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of ensemble.

        Parameters
        ----------
        x : torch.Tensor

        Returns
        -------
        torch.Tensor
            Ensemble logits or vote counts depending on strategy.
        """

        logits = self._collect_logits(x)

        strategy = self.config.strategy

        if strategy == "average":
            return self._average(logits)

        if strategy == "weighted_average":
            return self._weighted_average(logits)

        if strategy == "majority_vote":
            return self._majority_vote(logits)

        raise RuntimeError(f"Unsupported ensemble strategy: {strategy}")

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """
        Return ensemble probabilities.
        """

        logits = self.forward(x)

        if self.config.strategy == "majority_vote":
            probs = logits / logits.sum(dim=1, keepdim=True)
            return probs

        return torch.softmax(logits, dim=1)

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """
        Return predicted class labels.
        """

        probs = self.predict_proba(x)

        return torch.argmax(probs, dim=1)
