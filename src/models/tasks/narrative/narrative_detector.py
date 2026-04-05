"""
File Name: narrative_detector.py
Module: models.tasks.narrative
Description:
    Implements a transformer-based narrative detection model for the TruthLens AI
    system. The model detects narrative roles and narrative frame signals within
    text using contextual embeddings produced by a pretrained transformer
    encoder followed by a multi-label classification head.

    The dataset contains the following narrative labels:

        hero
        villain
        victim
        hero_entities
        villain_entities
        victim_entities
        RE
        HI
        CO
        MO
        EC

    These labels represent narrative actors and narrative frame indicators.
    Because multiple narrative signals may appear simultaneously in a single
    article, the model performs multi-label classification using sigmoid
    activation and BCEWithLogitsLoss.

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
    labels (optional): Tensor (batch_size, 11)
Outputs:
    Dictionary containing logits, probabilities, predictions, and optional loss
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, Optional, List

import torch
import torch.nn as nn

from ...base.base_model import BaseModel
from ...encoder.transformer_encoder import TransformerEncoder
from ...heads.multilabel_head import MultiLabelHead, MultiLabelHeadConfig
from ...training.trainer import Trainer, TrainerConfig

logger = logging.getLogger(__name__)


@dataclass
class NarrativeDetectorConfig:

    model_name: str = "roberta-base"
    pooling: str = "cls"
    dropout: float = 0.1
    threshold: float = 0.5
    device: Optional[str] = None


class NarrativeDetector(BaseModel):

    LABELS: List[str] = [
        "hero",
        "villain",
        "victim",
        "hero_entities",
        "villain_entities",
        "victim_entities",
        "RE",
        "HI",
        "CO",
        "MO",
        "EC",
    ]

    NUM_LABELS = len(LABELS)

    LABEL_MAPPING = {i: label for i, label in enumerate(LABELS)}

    def __init__(self, config: NarrativeDetectorConfig):

        super().__init__()

        self.config = config

        self.encoder = TransformerEncoder(
            model_name=config.model_name,
            pooling=config.pooling,
            device=config.device,
        )

        head_config = MultiLabelHeadConfig(
            input_dim=self.encoder.hidden_size,
            num_labels=self.NUM_LABELS,
            dropout=config.dropout,
            threshold=config.threshold,
        )

        self.classifier_head = MultiLabelHead(head_config)

        logger.info(
            "NarrativeDetector initialized | model=%s | labels=%d",
            config.model_name,
            self.NUM_LABELS,
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:

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
        threshold: Optional[float] = None,
    ) -> Dict[str, torch.Tensor]:

        self.eval()

        outputs = self.forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

        probabilities = outputs["probabilities"]

        thresh = threshold if threshold else self.config.threshold

        predictions = (probabilities >= thresh).int()

        return {
            "predictions": predictions,
            "probabilities": probabilities,
            "labels": self.LABEL_MAPPING,
        }

    def get_output_labels(self) -> Dict[int, str]:

        return self.LABEL_MAPPING

    def get_label_list(self) -> List[str]:

        return self.LABELS

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