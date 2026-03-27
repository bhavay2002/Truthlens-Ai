"""
File Name: ideology_classifier.py
Module: Model Architecture - Ideology Classification
Description:
    Implements a transformer-based ideology classification model used in the
    TruthLens AI system. The classifier predicts ideological orientation from
    textual input using contextual embeddings from a pretrained transformer
    encoder and a task-specific classification head.

    The module supports multi-class or multi-label ideology prediction and is
    designed for integration into larger analytical pipelines or multi-task
    systems.

Dependencies:
    logging
    typing
    torch
    torch.nn
    transformer_encoder (local module)

Inputs:
    Tokenized transformer inputs (input_ids, attention_mask)
    Optional ideology labels

Outputs:
    Dictionary containing logits, probabilities, and optional loss
"""

import logging
from typing import Dict, Optional

import torch
import torch.nn as nn

from src.models.encoder.transformer_encoder import TransformerEncoder


logger = logging.getLogger(__name__)


class IdeologyClassifier(nn.Module):
    """
    Transformer-based ideology classification model.
    """

    def __init__(
        self,
        encoder_model: str,
        num_labels: int,
        dropout: float = 0.1,
        device: Optional[str] = None,
    ) -> None:
        """Initialize ideology classifier."""

        super().__init__()

        if not isinstance(num_labels, int) or num_labels <= 0:
            raise ValueError("num_labels must be a positive integer")

        self.encoder = TransformerEncoder(
            model_name=encoder_model,
            pooling="cls",
            device=device,
        )

        hidden_size = self.encoder.hidden_size

        self.dropout = nn.Dropout(dropout)

        self.classifier = nn.Linear(hidden_size, num_labels)

        self.activation = nn.Sigmoid()

        self.device = self.encoder.device

        self.to(self.device)

        logger.info("IdeologyClassifier initialized")

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Run forward pass and optionally compute loss."""

        if input_ids is None or attention_mask is None:
            raise ValueError("input_ids and attention_mask cannot be None")

        encoder_outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

        pooled = encoder_outputs["pooled_output"]

        pooled = self.dropout(pooled)

        logits = self.classifier(pooled)

        probabilities = self.activation(logits)

        outputs: Dict[str, torch.Tensor] = {
            "logits": logits,
            "probabilities": probabilities,
        }

        if labels is not None:

            labels = labels.to(self.device)

            loss_fn = nn.BCEWithLogitsLoss()

            loss = loss_fn(logits, labels)

            outputs["loss"] = loss

        return outputs
