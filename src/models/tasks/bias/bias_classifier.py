"""
File Name: bias_classifier.py
Module: models.tasks.bias
Description:
    Implements a binary bias classification model used in the TruthLens AI system.
    The classifier predicts whether an article or text segment exhibits bias or
    is non-biased. The model uses a transformer encoder backbone (e.g., RoBERTa,
    BERT) followed by a classification head.

    This module is designed for production-grade ML pipelines and supports
    deterministic execution, structured outputs, and optional loss computation.

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


@dataclass
class BiasClassifierConfig:
    """
    Configuration for the BiasClassifier.
    """

    model_name: str = "roberta-base"
    pooling: str = "cls"
    dropout: float = 0.1
    device: Optional[str] = None


class BiasClassifier(nn.Module):
    """
    Binary bias classifier.

    Predicts:
        0 → non-bias
        1 → bias
    """

    NUM_CLASSES = 2

    def __init__(self, config: BiasClassifierConfig) -> None:
        super().__init__()

        if not isinstance(config, BiasClassifierConfig):
            raise TypeError("config must be BiasClassifierConfig")

        self.config = config

        self.encoder = TransformerEncoder(
            model_name=config.model_name,
            pooling=config.pooling,
            device=config.device,
        )

        head_config = ClassificationHeadConfig(
            input_dim=self.encoder.hidden_size,
            num_classes=self.NUM_CLASSES,
            dropout=config.dropout,
        )

        self.classifier_head = ClassificationHead(head_config)

        self.loss_fn = nn.CrossEntropyLoss()

        logger.info(
            "BiasClassifier initialized | model=%s | hidden=%d",
            config.model_name,
            self.encoder.hidden_size,
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
                Optional ground truth labels.

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

        logits = self.classifier_head(pooled_output)

        probabilities = F.softmax(logits, dim=-1)

        predictions = torch.argmax(probabilities, dim=-1)

        outputs: Dict[str, torch.Tensor] = {
            "logits": logits,
            "probabilities": probabilities,
            "predictions": predictions,
        }

        if labels is not None:

            if labels.dim() != 1:
                raise ValueError("labels must be 1D tensor (batch_size,)")

            loss = self.loss_fn(logits, labels)

            outputs["loss"] = loss

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
        Return label mapping.
        """

        return {
            0: "non_bias",
            1: "bias",
        }