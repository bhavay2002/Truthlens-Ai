"""
File Name: ideology_classifier.py
Module: models.tasks.ideology
Description:
    Implements a transformer-based ideology classification model used in the
    TruthLens AI system. The classifier predicts ideological orientation from
    textual input using contextual embeddings from a pretrained transformer
    encoder and a task-specific classification head.

    The dataset contains three labels representing political ideology:
        0 -> left
        1 -> center
        2 -> right

    The model performs multi-class classification using CrossEntropyLoss.

Author: TruthLens Engineering
Date: 2026-04-02
Dependencies:
    logging
    typing
    dataclasses
    torch
    torch.nn
    models.encoder.transformer_encoder
    models.heads.classification_head
Inputs:
    input_ids: Tensor (batch_size, sequence_length)
    attention_mask: Tensor (batch_size, sequence_length)
    labels (optional): Tensor (batch_size)
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

from ...encoder.transformer_encoder import TransformerEncoder
from ...heads.classification_head import (
    ClassificationHead,
    ClassificationHeadConfig,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

@dataclass
class IdeologyClassifierConfig:
    """
    Configuration for IdeologyClassifier.
    """

    model_name: str = "roberta-base"
    pooling: str = "cls"
    dropout: float = 0.1
    label_smoothing: float = 0.1
    device: Optional[str] = None


# ---------------------------------------------------------
# Ideology Classifier
# ---------------------------------------------------------

class IdeologyClassifier(nn.Module):
    """
    Transformer-based ideology classification model.

    Predicts:
        0 -> left
        1 -> center
        2 -> right
    """

    NUM_CLASSES = 3

    def __init__(self, config: IdeologyClassifierConfig) -> None:
        super().__init__()

        if not isinstance(config, IdeologyClassifierConfig):
            raise TypeError("config must be IdeologyClassifierConfig")

        self.config = config

        # -------------------------------------------------
        # Encoder
        # -------------------------------------------------

        self.encoder = TransformerEncoder(
            model_name=config.model_name,
            pooling=config.pooling,
            device=config.device,
        )

        # -------------------------------------------------
        # Classification Head
        # -------------------------------------------------

        head_config = ClassificationHeadConfig(
            input_dim=self.encoder.hidden_size,
            num_classes=self.NUM_CLASSES,
            dropout=config.dropout,
        )

        self.classifier_head = ClassificationHead(head_config)

        # -------------------------------------------------
        # Loss Function
        # -------------------------------------------------

        self.loss_fn = nn.CrossEntropyLoss(
            label_smoothing=config.label_smoothing
        )

        # -------------------------------------------------
        # Temperature Scaling (probability calibration)
        # -------------------------------------------------

        self.temperature = nn.Parameter(torch.ones(1))

        logger.info(
            "IdeologyClassifier initialized | model=%s | classes=%d",
            config.model_name,
            self.NUM_CLASSES,
        )

    # -----------------------------------------------------
    # Forward Pass
    # -----------------------------------------------------

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass.

        Args:
            input_ids:
                Token ids tensor (batch_size, seq_len)
            attention_mask:
                Attention mask tensor (batch_size, seq_len)
            labels:
                Optional ground truth labels

        Returns:
            Dictionary containing logits, probabilities, predictions,
            confidence scores, embeddings, and optional loss.
        """

        if input_ids is None or attention_mask is None:
            raise ValueError("input_ids and attention_mask must be provided")

        encoder_outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

        pooled_output = encoder_outputs["pooled_output"]

        logits = self.classifier_head(pooled_output)

        # Apply temperature scaling
        logits = logits / self.temperature

        probabilities = F.softmax(logits, dim=-1)

        predictions = torch.argmax(probabilities, dim=-1)

        confidence = torch.max(probabilities, dim=-1).values

        outputs: Dict[str, torch.Tensor] = {
            "logits": logits,
            "probabilities": probabilities,
            "predictions": predictions,
            "confidence": confidence,
            "embeddings": pooled_output,
        }

        # -------------------------------------------------
        # Loss computation (training)
        # -------------------------------------------------

        if labels is not None:

            if labels.dim() != 1:
                raise ValueError("labels must be 1D tensor (batch_size,)")

            loss = self.loss_fn(logits, labels.long())

            outputs["loss"] = loss

        return outputs

    # -----------------------------------------------------
    # Inference
    # -----------------------------------------------------

    @torch.inference_mode()
    def predict(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Run inference.

        Returns:
            predictions, probabilities, confidence
        """

        self.eval()

        outputs = self.forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

        return {
            "predictions": outputs["predictions"],
            "probabilities": outputs["probabilities"],
            "confidence": outputs["confidence"],
        }

    # -----------------------------------------------------
    # Label Mapping
    # -----------------------------------------------------

    def get_output_labels(self) -> Dict[int, str]:
        """
        Return ideology label mapping.
        """

        return {
            0: "left",
            1: "center",
            2: "right",
        }