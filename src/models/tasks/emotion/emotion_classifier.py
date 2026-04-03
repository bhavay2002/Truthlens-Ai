"""
File Name: emotion_classifier.py
Module: models.tasks.emotion
Description:
    Implements a multi-label emotion classification model for the TruthLens AI
    system. The classifier predicts multiple emotions simultaneously from
    text inputs using a transformer encoder backbone (e.g., RoBERTa, BERT)
    followed by a multi-label classification head.

    The dataset is assumed to contain 20 emotion labels:
        emotion_0 ... emotion_19

    The model outputs independent probabilities for each emotion using
    sigmoid activation and supports BCEWithLogitsLoss for training.

Author: TruthLens Engineering
Date: 2026-04-02
Dependencies:
    logging
    typing
    dataclasses
    torch
    torch.nn
    models.encoder.transformer_encoder
    models.heads.multilabel_head
Inputs:
    input_ids: Tensor (batch_size, sequence_length)
    attention_mask: Tensor (batch_size, sequence_length)
    labels (optional): Tensor (batch_size, 20)
Outputs:
    Dictionary containing logits, probabilities, predictions, and optional loss
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, Optional

import torch
import torch.nn as nn

from ...encoder.transformer_encoder import TransformerEncoder
from ...heads.multilabel_head import MultiLabelHead, MultiLabelHeadConfig

logger = logging.getLogger(__name__)


@dataclass
class EmotionClassifierConfig:
    """
    Configuration for EmotionClassifier.
    """

    model_name: str = "roberta-base"
    pooling: str = "cls"
    dropout: float = 0.1
    device: Optional[str] = None
    threshold: float = 0.5


class EmotionClassifier(nn.Module):
    """
    Multi-label emotion classifier.

    Predicts 20 independent emotion labels:
        emotion_0 ... emotion_19
    """

    NUM_EMOTIONS = 20

    def __init__(self, config: EmotionClassifierConfig) -> None:
        super().__init__()

        if not isinstance(config, EmotionClassifierConfig):
            raise TypeError("config must be EmotionClassifierConfig")

        self.config = config

        self.encoder = TransformerEncoder(
            model_name=config.model_name,
            pooling=config.pooling,
            device=config.device,
        )

        head_config = MultiLabelHeadConfig(
            input_dim=self.encoder.hidden_size,
            num_labels=self.NUM_EMOTIONS,
            dropout=config.dropout,
            threshold=config.threshold,
        )

        self.classifier_head = MultiLabelHead(head_config)

        logger.info(
            "EmotionClassifier initialized | model=%s | num_emotions=%d",
            config.model_name,
            self.NUM_EMOTIONS,
        )

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
                Token ids tensor.
            attention_mask:
                Attention mask tensor.
            labels:
                Optional multi-label tensor (batch_size, 20)

        Returns:
            Dictionary containing logits, probabilities, predictions,
            and optional loss.
        """

        if input_ids is None or attention_mask is None:
            raise ValueError("input_ids and attention_mask must be provided")

        encoder_outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

        pooled_output = encoder_outputs["pooled_output"]

        outputs = self.classifier_head(
            pooled_output,
            labels=labels,
        )

        return outputs

    @torch.no_grad()
    def predict(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Run inference.

        Returns:
            predictions and probabilities.
        """

        self.eval()

        outputs = self.forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

        return {
            "predictions": outputs["predictions"],
            "probabilities": outputs["probabilities"],
        }

    def get_output_labels(self) -> Dict[int, str]:
        """
        Return emotion label mapping.

        emotion_0 ... emotion_19
        """

        return {i: f"emotion_{i}" for i in range(self.NUM_EMOTIONS)}