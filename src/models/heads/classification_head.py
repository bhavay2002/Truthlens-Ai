"""
File Name: classification_head.py
Module: models.heads
Description:
    Implements a generic classification head used by downstream NLP models in
    the TruthLens AI system. The module defines a reusable neural network head
    that transforms encoder embeddings into classification logits. It supports
    dropout regularization, configurable hidden layers, and flexible activation
    functions.

    This head is designed to be attached to encoder outputs (e.g., transformer
    pooled embeddings) and is compatible with both single-task and multi-task
    architectures.

Dependencies:
    logging
    typing
    dataclasses
    torch
    torch.nn
Inputs:
    Encoder embeddings (batch_size, hidden_dim)
Outputs:
    Classification logits (batch_size, num_classes)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn


logger = logging.getLogger(__name__)


@dataclass
class ClassificationHeadConfig:
    """
    Configuration for the classification head.
    """

    input_dim: int
    num_classes: int
    hidden_dim: Optional[int] = None
    dropout: float = 0.1
    activation: str = "gelu"


class ClassificationHead(nn.Module):
    """
    Generic classification head for encoder-based models.

    The head optionally contains a hidden projection layer before the final
    classifier. This improves representation learning for complex tasks.
    """

    SUPPORTED_ACTIVATIONS = {
        "relu": nn.ReLU,
        "gelu": nn.GELU,
        "tanh": nn.Tanh,
    }

    def __init__(self, config: ClassificationHeadConfig) -> None:
        """
        Initialize classification head.

        Args:
            config:
                Configuration object describing head parameters.
        """

        super().__init__()

        if config.input_dim <= 0:
            raise ValueError("input_dim must be positive")

        if config.num_classes <= 0:
            raise ValueError("num_classes must be positive")

        if config.dropout < 0 or config.dropout > 1:
            raise ValueError("dropout must be between 0 and 1")

        if config.activation not in self.SUPPORTED_ACTIVATIONS:
            raise ValueError(
                f"Unsupported activation '{config.activation}'. "
                f"Supported: {list(self.SUPPORTED_ACTIVATIONS.keys())}"
            )

        self.config = config

        activation_cls = self.SUPPORTED_ACTIVATIONS[config.activation]

        layers = []

        if config.hidden_dim:
            if config.hidden_dim <= 0:
                raise ValueError("hidden_dim must be positive if provided")

            layers.extend(
                [
                    nn.Linear(config.input_dim, config.hidden_dim),
                    activation_cls(),
                    nn.Dropout(config.dropout),
                    nn.Linear(config.hidden_dim, config.num_classes),
                ]
            )
        else:
            layers.extend(
                [
                    nn.Dropout(config.dropout),
                    nn.Linear(config.input_dim, config.num_classes),
                ]
            )

        self.classifier = nn.Sequential(*layers)

        logger.info(
            "ClassificationHead initialized | input_dim=%d | num_classes=%d",
            config.input_dim,
            config.num_classes,
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the classification head.

        Args:
            features:
                Tensor of shape (batch_size, input_dim)

        Returns:
            Logits tensor (batch_size, num_classes)
        """

        if features is None:
            raise ValueError("features cannot be None")

        if features.dim() != 2:
            raise ValueError(
                f"Expected 2D tensor (batch_size, input_dim), got {features.shape}"
            )

        logits = self.classifier(features)

        return logits

    def get_output_dim(self) -> int:
        """
        Returns number of output classes.
        """

        return self.config.num_classes