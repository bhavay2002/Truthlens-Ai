"""
File Name: regression_head.py
Module: models.heads
Description:
    Implements a reusable regression head for encoder-based models in the
    TruthLens AI system. The module defines a neural network projection head
    that maps encoder embeddings into continuous regression outputs.

    It supports configurable hidden layers, dropout regularization, activation
    functions, and output dimensionality. This head can be used for tasks such
    as credibility scoring, risk estimation, sentiment intensity prediction,
    and other scalar or vector regression objectives.

Dependencies:
    logging
    typing
    dataclasses
    torch
    torch.nn
Inputs:
    Encoder embeddings (batch_size, hidden_dim)
Outputs:
    Regression outputs (batch_size, output_dim)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


@dataclass
class RegressionHeadConfig:
    """
    Configuration for regression head.
    """

    input_dim: int
    output_dim: int = 1
    hidden_dim: Optional[int] = None
    dropout: float = 0.1
    activation: str = "gelu"


class RegressionHead(nn.Module):
    """
    Generic regression head for encoder-based architectures.

    This module transforms encoder embeddings into continuous outputs.
    It supports optional intermediate projection layers for increased
    modeling capacity.
    """

    SUPPORTED_ACTIVATIONS = {
        "relu": nn.ReLU,
        "gelu": nn.GELU,
        "tanh": nn.Tanh,
        "elu": nn.ELU,
    }

    def __init__(self, config: RegressionHeadConfig) -> None:
        """
        Initialize regression head.

        Args:
            config:
                Configuration describing regression head structure.
        """

        super().__init__()

        if config.input_dim <= 0:
            raise ValueError("input_dim must be positive")

        if config.output_dim <= 0:
            raise ValueError("output_dim must be positive")

        if config.dropout < 0 or config.dropout > 1:
            raise ValueError("dropout must be between 0 and 1")

        if config.activation not in self.SUPPORTED_ACTIVATIONS:
            raise ValueError(
                f"Unsupported activation '{config.activation}'. "
                f"Supported: {list(self.SUPPORTED_ACTIVATIONS.keys())}"
            )

        self.config = config
        activation_cls = self.SUPPORTED_ACTIVATIONS[config.activation]

        layers: list[nn.Module] = []

        if config.hidden_dim is not None:
            if config.hidden_dim <= 0:
                raise ValueError("hidden_dim must be positive if provided")

            layers.extend(
                [
                    nn.Linear(config.input_dim, config.hidden_dim),
                    activation_cls(),
                    nn.Dropout(config.dropout),
                    nn.Linear(config.hidden_dim, config.output_dim),
                ]
            )
        else:
            layers.extend(
                [
                    nn.Dropout(config.dropout),
                    nn.Linear(config.input_dim, config.output_dim),
                ]
            )

        self.regressor = nn.Sequential(*layers)

        logger.info(
            "RegressionHead initialized | input_dim=%d | output_dim=%d",
            config.input_dim,
            config.output_dim,
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for regression head.

        Args:
            features:
                Encoder embeddings of shape (batch_size, input_dim)

        Returns:
            Tensor of shape (batch_size, output_dim)
        """

        if features is None:
            raise ValueError("features cannot be None")

        if features.dim() != 2:
            raise ValueError(
                f"Expected features shape (batch_size, input_dim), got {features.shape}"
            )

        outputs = self.regressor(features)

        return outputs

    def get_output_dim(self) -> int:
        """
        Returns regression output dimensionality.
        """

        return self.config.output_dim