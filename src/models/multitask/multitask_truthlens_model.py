"""
File Name: multitask_truthlens_model.py
Module: Model Architecture - Multi-Task TruthLens Model
Description:
    Defines the core multi-task neural architecture used in the TruthLens AI
    system. The model uses a shared transformer encoder and multiple task-
    specific classification heads for tasks such as bias detection, emotion
    classification, propaganda detection, stance detection, and misinformation
    analysis.

    The design follows modern multi-task NLP research practices where a shared
    language representation is learned by a common encoder while specialized
    heads handle individual tasks.

Dependencies:
    logging
    typing
    torch
    torch.nn
    transformer_encoder (local module)

Inputs:
    Tokenized transformer inputs (input_ids, attention_mask)
    Optional task labels

Outputs:
    Dictionary containing logits, probabilities, and optional losses for tasks
"""

import logging
from typing import Dict, Optional

import torch
import torch.nn as nn

from transformer_encoder import TransformerEncoder


logger = logging.getLogger(__name__)


class TaskHead(nn.Module):
    """
    Generic task-specific classification head.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        dropout: float = 0.1,
    ) -> None:
        """Initialize classification head."""

        super().__init__()

        if input_dim <= 0 or output_dim <= 0:
            raise ValueError("input_dim and output_dim must be positive")

        self.dropout = nn.Dropout(dropout)

        self.classifier = nn.Linear(input_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through classification head."""

        x = self.dropout(x)

        logits = self.classifier(x)

        return logits


class MultiTaskTruthLensModel(nn.Module):
    """
    Multi-task architecture combining shared encoder with multiple heads.
    """

    def __init__(
        self,
        encoder_model: str,
        num_bias_labels: int,
        num_emotion_labels: int,
        num_propaganda_labels: int,
        num_stance_labels: int,
        num_misinformation_labels: int,
        dropout: float = 0.1,
        device: Optional[str] = None,
    ) -> None:
        """Initialize multi-task TruthLens model."""

        super().__init__()

        self.encoder = TransformerEncoder(
            model_name=encoder_model,
            pooling="cls",
            device=device,
        )

        hidden_size = self.encoder.hidden_size

        self.bias_head = TaskHead(hidden_size, num_bias_labels, dropout)

        self.emotion_head = TaskHead(hidden_size, num_emotion_labels, dropout)

        self.propaganda_head = TaskHead(hidden_size, num_propaganda_labels, dropout)

        self.stance_head = TaskHead(hidden_size, num_stance_labels, dropout)

        self.misinformation_head = TaskHead(
            hidden_size,
            num_misinformation_labels,
            dropout,
        )

        self.device = self.encoder.device

        self.to(self.device)

        logger.info("MultiTaskTruthLensModel initialized")

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Dict[str, torch.Tensor]:
        """Forward pass through encoder and all task heads."""

        if input_ids is None or attention_mask is None:
            raise ValueError("input_ids and attention_mask cannot be None")

        encoder_outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

        pooled = encoder_outputs["pooled_output"]

        bias_logits = self.bias_head(pooled)
        emotion_logits = self.emotion_head(pooled)
        propaganda_logits = self.propaganda_head(pooled)
        stance_logits = self.stance_head(pooled)
        misinformation_logits = self.misinformation_head(pooled)

        outputs: Dict[str, torch.Tensor] = {
            "bias_logits": bias_logits,
            "emotion_logits": emotion_logits,
            "propaganda_logits": propaganda_logits,
            "stance_logits": stance_logits,
            "misinformation_logits": misinformation_logits,
        }

        if labels is not None:

            loss_fn = nn.BCEWithLogitsLoss()

            losses = []

            if "bias" in labels:
                losses.append(
                    loss_fn(bias_logits, labels["bias"].to(self.device))
                )

            if "emotion" in labels:
                losses.append(
                    loss_fn(emotion_logits, labels["emotion"].to(self.device))
                )

            if "propaganda" in labels:
                losses.append(
                    loss_fn(
                        propaganda_logits,
                        labels["propaganda"].to(self.device),
                    )
                )

            if "stance" in labels:
                losses.append(
                    loss_fn(
                        stance_logits,
                        labels["stance"].to(self.device),
                    )
                )

            if "misinformation" in labels:
                losses.append(
                    loss_fn(
                        misinformation_logits,
                        labels["misinformation"].to(self.device),
                    )
                )

            if losses:
                total_loss = torch.stack(losses).mean()
                outputs["loss"] = total_loss

        return outputs