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
from typing import Any, Dict, Optional

import torch
import torch.nn as nn

from ...base.base_model import BaseModel
from ...encoder.transformer_encoder import TransformerEncoder
from ...heads.multilabel_head import MultiLabelHead, MultiLabelHeadConfig
from ...training.trainer import Trainer, TrainerConfig
from src.features.emotion.emotion_schema import EMOTION_LABELS

logger = logging.getLogger(__name__)


# ------------------------------------------------------------
# Configuration
# ------------------------------------------------------------

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


# ------------------------------------------------------------
# Emotion Classifier
# ------------------------------------------------------------

class EmotionClassifier(BaseModel):
    """
    Multi-label emotion classifier.

    Dataset labels:
        emotion0 ... emotion19

    Emotion names are mapped from EMOTION_LABELS.
    """

    NUM_EMOTIONS = 20

    def __init__(self, config: EmotionClassifierConfig) -> None:
        super().__init__()

        if not isinstance(config, EmotionClassifierConfig):
            raise TypeError("config must be EmotionClassifierConfig")

        self.config = config

        # ------------------------------------------------
        # Encoder
        # ------------------------------------------------

        self.encoder = TransformerEncoder(
            model_name=config.model_name,
            pooling=config.pooling,
            device=config.device,
        )

        # ------------------------------------------------
        # Classification head
        # ------------------------------------------------

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

    # ------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:

        if input_ids is None or attention_mask is None:
            raise ValueError("input_ids and attention_mask must be provided")

        if labels is not None and labels.shape[-1] != self.NUM_EMOTIONS:
            raise ValueError(
                f"labels must have shape (batch_size, {self.NUM_EMOTIONS})"
            )

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

    # ------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------

    @torch.inference_mode()
    def predict(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:

        self.eval()

        outputs = self.forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

        return {
            "predictions": outputs["predictions"],
            "probabilities": outputs["probabilities"],
        }

    # ------------------------------------------------------------
    # Label Mapping
    # ------------------------------------------------------------

    def get_output_labels(self) -> Dict[int, str]:
        """
        Map numeric output indices to emotion names.
        """

        return {i: EMOTION_LABELS[i] for i in range(self.NUM_EMOTIONS)}

    def get_training_labels(self) -> Dict[int, str]:
        """
        Return dataset label names (emotion0…emotion19).
        """

        return {i: f"emotion{i}" for i in range(self.NUM_EMOTIONS)}

    def create_trainer(
        self,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[Any] = None,
        config: Optional[TrainerConfig] = None,
    ) -> Trainer:
        """
        Build a TruthLensTrainer for this model.
        """
        from dataclasses import replace as _replace

        effective_config = config if config is not None else TrainerConfig()
        effective_config = _replace(
            effective_config,
            architecture=type(self).__name__,
            model_name=self.config.model_name,
        )
        return Trainer(
            model=self,
            optimizer=optimizer,
            scheduler=scheduler,
            config=effective_config,
        )