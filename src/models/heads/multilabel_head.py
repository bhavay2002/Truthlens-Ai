"""
File Name: multilabel_head.py
Module: models.heads
Description:
    Implements a reusable multi-label classification head for encoder-based
    architectures in the TruthLens AI system. This head projects encoder
    embeddings into independent label logits suitable for multi-label tasks
    (e.g., emotion classification, propaganda technique detection).

    The module supports optional hidden projection layers, configurable
    activation functions, dropout regularization, and built-in loss handling
    using BCEWithLogitsLoss. It returns logits, probabilities, and binary
    predictions for downstream pipelines.

Dependencies:
    logging
    typing
    dataclasses
    torch
    torch.nn
Inputs:
    Encoder embeddings (batch_size, input_dim)
Outputs:
    Dictionary containing logits, probabilities, predictions, and optional loss
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


@dataclass
class MultiLabelHeadConfig:
    """
    Configuration for the multi-label classification head.
    """

    input_dim: int
    num_labels: int
    hidden_dim: Optional[int] = None
    dropout: float = 0.1
    activation: str = "gelu"
    threshold: float = 0.5


class MultiLabelHead(nn.Module):
    """
    Multi-label classification head.

    Converts encoder embeddings into independent label logits using
    sigmoid activation for probability estimation.
    """

    SUPPORTED_ACTIVATIONS = {
        "relu": nn.ReLU,
        "gelu": nn.GELU,
        "tanh": nn.Tanh,
        "elu": nn.ELU,
    }

    def __init__(self, config: MultiLabelHeadConfig) -> None:
        """
        Initialize the multi-label classification head.

        Args:
            config:
                MultiLabelHeadConfig containing architecture parameters.
        """

        super().__init__()

        if config.input_dim <= 0:
            raise ValueError("input_dim must be positive")

        if config.num_labels <= 0:
            raise ValueError("num_labels must be positive")

        if config.dropout < 0 or config.dropout > 1:
            raise ValueError("dropout must be between 0 and 1")

        if config.activation not in self.SUPPORTED_ACTIVATIONS:
            raise ValueError(
                f"Unsupported activation '{config.activation}'. "
                f"Supported: {list(self.SUPPORTED_ACTIVATIONS.keys())}"
            )

        if not (0 < config.threshold < 1):
            raise ValueError("threshold must be between 0 and 1")

        self.config = config

        activation_cls = self.SUPPORTED_ACTIVATIONS[config.activation]

        layers: list[nn.Module] = []

        if config.hidden_dim:
            if config.hidden_dim <= 0:
                raise ValueError("hidden_dim must be positive")

            layers.extend(
                [
                    nn.Linear(config.input_dim, config.hidden_dim),
                    activation_cls(),
                    nn.Dropout(config.dropout),
                    nn.Linear(config.hidden_dim, config.num_labels),
                ]
            )
        else:
            layers.extend(
                [
                    nn.Dropout(config.dropout),
                    nn.Linear(config.input_dim, config.num_labels),
                ]
            )

        self.classifier = nn.Sequential(*layers)

        self.loss_fn = nn.BCEWithLogitsLoss()

        logger.info(
            "MultiLabelHead initialized | input_dim=%d | num_labels=%d",
            config.input_dim,
            config.num_labels,
        )

    def forward(
        self,
        features: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass for multi-label classification.

        Args:
            features:
                Encoder embeddings (batch_size, input_dim)
            labels:
                Optional ground truth labels (batch_size, num_labels)

        Returns:
            Dictionary containing logits, probabilities, predictions,
            and optional loss.
        """

        if features is None:
            raise ValueError("features cannot be None")

        if features.dim() != 2:
            raise ValueError(
                f"Expected features shape (batch_size, input_dim), got {features.shape}"
            )

        logits = self.classifier(features)

        probabilities = torch.sigmoid(logits)

        predictions = (probabilities >= self.config.threshold).long()

        outputs: Dict[str, torch.Tensor] = {
            "logits": logits,
            "probabilities": probabilities,
            "predictions": predictions,
        }

        if labels is not None:
            if labels.shape != logits.shape:
                raise ValueError(
                    f"labels shape must match logits shape. "
                    f"Expected {logits.shape}, got {labels.shape}"
                )

            loss = self.loss_fn(logits, labels.float())
            outputs["loss"] = loss

        return outputs

    def get_output_dim(self) -> int:
        """
        Returns number of labels predicted by the head.
        """

        return self.config.num_labels