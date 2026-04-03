"""
File Name: multitask_truthlens_model.py
Module: models.multitask
Description:
    Defines the core multi-task neural architecture used in the TruthLens AI
    system. The model uses a shared transformer encoder and multiple task-
    specific heads for tasks including:

        - bias detection (binary)
        - ideology classification (left/center/right)
        - propaganda detection (binary)
        - narrative role detection (hero/villain/victim)
        - narrative frame detection (RE/HI/CO/MO/EC)
        - emotion classification (20-label multi-label)

    The architecture follows modern multi-task NLP research practices where
    a shared contextual encoder learns a universal representation while
    task-specific heads specialize for downstream objectives.

Dependencies:
    logging
    typing
    dataclasses
    torch
    torch.nn
    models.encoder.transformer_encoder
    models.heads.classification_head
    models.heads.multilabel_head
Inputs:
    input_ids: Tensor (batch_size, sequence_length)
    attention_mask: Tensor (batch_size, sequence_length)
    labels (optional): Dict[str, Tensor]
Outputs:
    Dictionary containing logits, probabilities, predictions, and optional loss
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, Optional, Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..encoder.transformer_encoder import TransformerEncoder
from ..heads.classification_head import ClassificationHead, ClassificationHeadConfig
from ..heads.multilabel_head import MultiLabelHead, MultiLabelHeadConfig


logger = logging.getLogger(__name__)


@dataclass
class MultiTaskTruthLensConfig:
    """
    Configuration for the multi-task TruthLens model.
    """

    model_name: str = "roberta-base"
    pooling: str = "cls"
    dropout: float = 0.1
    device: Optional[str] = None


class MultiTaskTruthLensModel(nn.Module):
    """
    Multi-task TruthLens architecture.
    """

    NUM_BIAS = 2
    NUM_IDEOLOGY = 3
    NUM_PROPAGANDA = 2
    NUM_NARRATIVE = 3
    NUM_NARRATIVE_FRAMES = 5
    NUM_EMOTIONS = 20

    def __init__(self, config: MultiTaskTruthLensConfig) -> None:
        super().__init__()

        if not isinstance(config, MultiTaskTruthLensConfig):
            raise TypeError("config must be MultiTaskTruthLensConfig")

        self.config = config

        self.encoder = TransformerEncoder(
            model_name=config.model_name,
            pooling=config.pooling,
            device=config.device,
        )

        hidden = self.encoder.hidden_size

        # Classification heads
        self.bias_head = ClassificationHead(
            ClassificationHeadConfig(hidden, self.NUM_BIAS, dropout=config.dropout)
        )

        self.ideology_head = ClassificationHead(
            ClassificationHeadConfig(hidden, self.NUM_IDEOLOGY, dropout=config.dropout)
        )

        self.propaganda_head = ClassificationHead(
            ClassificationHeadConfig(hidden, self.NUM_PROPAGANDA, dropout=config.dropout)
        )

        # Multi-label heads
        self.narrative_head = MultiLabelHead(
            MultiLabelHeadConfig(hidden, self.NUM_NARRATIVE, dropout=config.dropout)
        )

        self.narrative_frame_head = MultiLabelHead(
            MultiLabelHeadConfig(hidden, self.NUM_NARRATIVE_FRAMES, dropout=config.dropout)
        )

        self.emotion_head = MultiLabelHead(
            MultiLabelHeadConfig(hidden, self.NUM_EMOTIONS, dropout=config.dropout)
        )

        self.loss_ce = nn.CrossEntropyLoss()
        self.loss_bce = nn.BCEWithLogitsLoss()

        logger.info("MultiTaskTruthLensModel initialized")

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Dict[str, Any]:

        if input_ids is None or attention_mask is None:
            raise ValueError("input_ids and attention_mask cannot be None")

        encoder_outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

        pooled = encoder_outputs["pooled_output"]

        bias_logits = self.bias_head(pooled)
        ideology_logits = self.ideology_head(pooled)
        propaganda_logits = self.propaganda_head(pooled)

        narrative_outputs = self.narrative_head(pooled)
        narrative_frame_outputs = self.narrative_frame_head(pooled)
        emotion_outputs = self.emotion_head(pooled)

        outputs: Dict[str, Any] = {
            "bias_logits": bias_logits,
            "ideology_logits": ideology_logits,
            "propaganda_logits": propaganda_logits,
            "narrative_logits": narrative_outputs["logits"],
            "narrative_frame_logits": narrative_frame_outputs["logits"],
            "emotion_logits": emotion_outputs["logits"],
            "bias_probabilities": F.softmax(bias_logits, dim=-1),
            "ideology_probabilities": F.softmax(ideology_logits, dim=-1),
            "propaganda_probabilities": F.softmax(propaganda_logits, dim=-1),
            "narrative_probabilities": narrative_outputs["probabilities"],
            "narrative_frame_probabilities": narrative_frame_outputs["probabilities"],
            "emotion_probabilities": emotion_outputs["probabilities"],
        }

        if labels is not None:

            task_losses: Dict[str, torch.Tensor] = {}

            if "bias" in labels:
                task_losses["bias"] = self.loss_ce(
                    bias_logits,
                    labels["bias"].long(),
                )

            if "ideology" in labels:
                task_losses["ideology"] = self.loss_ce(
                    ideology_logits,
                    labels["ideology"].long(),
                )

            if "propaganda" in labels:
                task_losses["propaganda"] = self.loss_ce(
                    propaganda_logits,
                    labels["propaganda"].long(),
                )

            if "narrative" in labels:
                task_losses["narrative"] = self.loss_bce(
                    narrative_outputs["logits"],
                    labels["narrative"].float(),
                )

            if "narrative_frame" in labels:
                task_losses["narrative_frame"] = self.loss_bce(
                    narrative_frame_outputs["logits"],
                    labels["narrative_frame"].float(),
                )

            if "emotion" in labels:
                task_losses["emotion"] = self.loss_bce(
                    emotion_outputs["logits"],
                    labels["emotion"].float(),
                )

            if task_losses:
                total_loss = torch.stack(list(task_losses.values())).mean()
                outputs["loss"] = total_loss
                outputs["task_losses"] = task_losses

        return outputs