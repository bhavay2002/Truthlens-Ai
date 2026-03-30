"""
File Name: propaganda_detector.py
Module: Model Architecture - Propaganda Detection
Description:
    Implements a transformer-based propaganda detection model for the
    TruthLens AI system. The model predicts propaganda techniques or
    propaganda presence within text using contextual embeddings generated
    by a pretrained transformer encoder and a classification head.

    Supports configurable task modes:
    - binary (default; suitable for No/Yes)
    - multi_class
    - multi_label

Dependencies:
    logging
    typing
    torch
    torch.nn
    transformer_encoder (local module)

Inputs:
    Tokenized transformer inputs (input_ids, attention_mask)
    Optional propaganda labels

Outputs:
    Dictionary containing logits, probabilities, and optional loss
"""

import logging
from typing import Dict, Optional

import torch
import torch.nn as nn

from src.models.encoder.transformer_encoder import TransformerEncoder


logger = logging.getLogger(__name__)


class PropagandaDetector(nn.Module):
    """
    Transformer-based propaganda detection model.
    """

    def __init__(
        self,
        encoder_model: str,
        num_labels: int,
        task_type: str = "binary",
        dropout: float = 0.1,
        device: Optional[str] = None,
    ) -> None:
        """Initialize propaganda detection model."""

        super().__init__()

        if not isinstance(num_labels, int) or num_labels <= 0:
            raise ValueError("num_labels must be a positive integer")

        if task_type not in {"binary", "multi_class", "multi_label"}:
            raise ValueError(
                "task_type must be one of: 'binary', 'multi_class', 'multi_label'"
            )

        self.encoder = TransformerEncoder(
            model_name=encoder_model,
            pooling="cls",
            device=device,
        )

        hidden_size = self.encoder.hidden_size

        self.dropout = nn.Dropout(dropout)

        self.classifier = nn.Linear(hidden_size, num_labels)

        self.task_type = task_type
        self.num_labels = num_labels

        if self.task_type == "multi_label":
            self.activation = nn.Sigmoid()
        elif self.task_type == "binary" and self.num_labels == 1:
            self.activation = nn.Sigmoid()
        else:
            self.activation = nn.Softmax(dim=-1)

        self.device = self.encoder.device

        self.to(self.device)

        logger.info("PropagandaDetector initialized")

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

            if self.task_type == "multi_label":
                loss_fn = nn.BCEWithLogitsLoss()
                loss = loss_fn(logits, labels.float())
            elif self.task_type == "binary" and self.num_labels == 1:
                loss_fn = nn.BCEWithLogitsLoss()
                if labels.dim() == 1:
                    labels = labels.unsqueeze(1)
                loss = loss_fn(logits, labels.float())
            else:
                loss_fn = nn.CrossEntropyLoss()
                if labels.dim() == 2 and labels.size(1) == self.num_labels:
                    labels = labels.argmax(dim=1)
                loss = loss_fn(logits, labels.long())

            outputs["loss"] = loss

        return outputs
