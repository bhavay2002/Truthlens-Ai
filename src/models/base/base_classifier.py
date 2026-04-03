"""
File Name: base_classifier.py
Module: models.base
Description:
    Provides a reusable base class for classification models in the TruthLens ML
    framework. This module extends the BaseModel abstraction with functionality
    specific to classification tasks, including logits generation, probability
    computation, prediction utilities, and loss handling. It is designed to work
    with PyTorch-based architectures and supports both single-label and
    multi-label classification tasks.

Dependencies:
    torch
    torch.nn
    torch.nn.functional
    typing
    logging
    models.base.base_model
Inputs:
    Tensor inputs representing encoded features or embeddings.
Outputs:
    Logits, probabilities, predicted labels, and loss values.
"""

from __future__ import annotations

import logging
from abc import abstractmethod
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn

from .base_model import BaseModel

logger = logging.getLogger(__name__)


class BaseClassifier(BaseModel):
    """
    Base class for classification models.

    This abstraction provides common functionality for classification-based
    neural networks, including prediction helpers, probability computation,
    and loss calculation. Specific classifier implementations must implement
    the `encode` method to generate feature representations.
    """

    def __init__(
        self,
        num_classes: int,
        input_dim: int,
        dropout: float = 0.1,
        multi_label: bool = False,
    ) -> None:
        """
        Initializes the classifier.

        Args:
            num_classes: Number of output classes.
            input_dim: Dimension of input feature representation.
            dropout: Dropout probability.
            multi_label: Whether the task is multi-label classification.
        """
        super().__init__()

        if num_classes <= 0:
            raise ValueError("num_classes must be positive")

        if input_dim <= 0:
            raise ValueError("input_dim must be positive")

        self.num_classes = num_classes
        self.input_dim = input_dim
        self.multi_label = multi_label

        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(input_dim, num_classes)

        if multi_label:
            self.loss_fn: nn.Module = nn.BCEWithLogitsLoss()
        else:
            self.loss_fn = nn.CrossEntropyLoss()

    @abstractmethod
    def encode(self, *inputs: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        """
        Encodes raw inputs into a feature representation.

        This method must be implemented by subclasses.

        Returns:
            Tensor of shape (batch_size, input_dim)
        """
        raise NotImplementedError("Subclasses must implement encode().")

    def forward(
        self,
        *inputs: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        **kwargs: Any,
    ) -> Dict[str, torch.Tensor]:
        """
        Executes a forward pass.

        Args:
            *inputs: Input tensors.
            labels: Optional labels for computing loss.

        Returns:
            Dictionary containing logits, probabilities, predictions,
            and optionally loss.
        """
        features = self.encode(*inputs, **kwargs)

        if features.dim() != 2:
            raise ValueError(
                f"Encoded features must be 2D (batch_size, feature_dim), "
                f"got shape {features.shape}"
            )

        features = self.dropout(features)
        logits = self.classifier(features)

        output: Dict[str, torch.Tensor] = {"logits": logits}

        if self.multi_label:
            probs = torch.sigmoid(logits)
            preds = (probs > 0.5).long()
        else:
            probs = F.softmax(logits, dim=-1)
            preds = torch.argmax(probs, dim=-1)

        output["probabilities"] = probs
        output["predictions"] = preds

        if labels is not None:
            loss = self.compute_loss(logits, labels)
            output["loss"] = loss

        return output

    def compute_loss(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """
        Computes classification loss.

        Args:
            logits: Model output logits.
            labels: Ground truth labels.

        Returns:
            Loss tensor.
        """
        if self.multi_label:
            labels = labels.float()

        loss = self.loss_fn(logits, labels)
        return loss

    @torch.no_grad()
    def predict(
        self,
        *inputs: torch.Tensor,
        **kwargs: Any,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Performs inference and returns predictions and probabilities.

        Args:
            *inputs: Input tensors.

        Returns:
            Tuple (predictions, probabilities)
        """
        self.eval()

        outputs = self.forward(*inputs, **kwargs)

        predictions = outputs["predictions"]
        probabilities = outputs["probabilities"]

        return predictions, probabilities

    @torch.no_grad()
    def predict_logits(
        self,
        *inputs: torch.Tensor,
        **kwargs: Any,
    ) -> torch.Tensor:
        """
        Returns raw logits for inference.

        Args:
            *inputs: Input tensors.

        Returns:
            Logits tensor.
        """
        self.eval()
        outputs = self.forward(*inputs, **kwargs)
        return outputs["logits"]

    def get_config(self) -> Dict[str, Any]:
        """
        Returns classifier configuration.

        Returns:
            Dictionary containing model configuration.
        """
        config = {
            "num_classes": self.num_classes,
            "input_dim": self.input_dim,
            "multi_label": self.multi_label,
        }

        logger.debug("Classifier config: %s", config)
        return config