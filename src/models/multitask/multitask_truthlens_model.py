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


# ------------------------------------------------------------
# Configuration
# ------------------------------------------------------------

@dataclass
class MultiTaskTruthLensConfig:

    model_name: str = "roberta-base"
    pooling: str = "cls"
    dropout: float = 0.1
    device: Optional[str] = None

    bias_weight: float = 1.0
    ideology_weight: float = 1.0
    propaganda_weight: float = 1.0
    narrative_weight: float = 1.0
    narrative_frame_weight: float = 1.0
    emotion_weight: float = 1.0


# ------------------------------------------------------------
# Model
# ------------------------------------------------------------

class MultiTaskTruthLensModel(nn.Module):

    # Label definitions
    BIAS_LABELS = ["non_bias", "bias"]
    IDEOLOGY_LABELS = ["left", "center", "right"]
    PROPAGANDA_LABELS = ["non_propaganda", "propaganda"]

    NARRATIVE_LABELS = ["hero", "villain", "victim"]

    FRAME_LABELS = ["RE", "HI", "CO", "MO", "EC"]

    NUM_BIAS = len(BIAS_LABELS)
    NUM_IDEOLOGY = len(IDEOLOGY_LABELS)
    NUM_PROPAGANDA = len(PROPAGANDA_LABELS)
    NUM_NARRATIVE = len(NARRATIVE_LABELS)
    NUM_NARRATIVE_FRAMES = len(FRAME_LABELS)

    NUM_EMOTIONS = 20

    def __init__(self, config: MultiTaskTruthLensConfig):

        super().__init__()

        self.config = config

        # ----------------------------------------------------
        # Shared Encoder
        # ----------------------------------------------------

        self.encoder = TransformerEncoder(
            model_name=config.model_name,
            pooling=config.pooling,
            device=config.device,
        )

        hidden = self.encoder.hidden_size

        # ----------------------------------------------------
        # Task Heads
        # ----------------------------------------------------

        self.bias_head = ClassificationHead(
            ClassificationHeadConfig(hidden, self.NUM_BIAS, dropout=config.dropout)
        )

        self.ideology_head = ClassificationHead(
            ClassificationHeadConfig(hidden, self.NUM_IDEOLOGY, dropout=config.dropout)
        )

        self.propaganda_head = ClassificationHead(
            ClassificationHeadConfig(hidden, self.NUM_PROPAGANDA, dropout=config.dropout)
        )

        self.narrative_head = MultiLabelHead(
            MultiLabelHeadConfig(hidden, self.NUM_NARRATIVE, dropout=config.dropout)
        )

        self.narrative_frame_head = MultiLabelHead(
            MultiLabelHeadConfig(hidden, self.NUM_NARRATIVE_FRAMES, dropout=config.dropout)
        )

        self.emotion_head = MultiLabelHead(
            MultiLabelHeadConfig(hidden, self.NUM_EMOTIONS, dropout=config.dropout)
        )

        # ----------------------------------------------------
        # Loss functions
        # ----------------------------------------------------

        self.loss_ce = nn.CrossEntropyLoss(label_smoothing=0.1)
        self.loss_bce = nn.BCEWithLogitsLoss()

        # temperature scaling
        self.temperature = nn.Parameter(torch.ones(1))

        logger.info("MultiTaskTruthLensModel initialized")

    # ------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Dict[str, Any]:

        encoder_outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

        pooled = encoder_outputs["pooled_output"]

        # stabilize temperature
        temperature = torch.clamp(self.temperature, 0.5, 5.0)

        # ----------------------------------------------------
        # Task heads
        # ----------------------------------------------------

        bias_logits = self.bias_head(pooled) / temperature
        ideology_logits = self.ideology_head(pooled) / temperature
        propaganda_logits = self.propaganda_head(pooled) / temperature

        narrative_outputs = self.narrative_head(pooled)
        narrative_frame_outputs = self.narrative_frame_head(pooled)
        emotion_outputs = self.emotion_head(pooled)

        bias_probs = F.softmax(bias_logits, dim=-1)
        ideology_probs = F.softmax(ideology_logits, dim=-1)
        propaganda_probs = F.softmax(propaganda_logits, dim=-1)

        outputs: Dict[str, Any] = {

            "embeddings": pooled,

            "bias": {
                "logits": bias_logits,
                "probabilities": bias_probs,
                "predictions": torch.argmax(bias_probs, dim=-1),
            },

            "ideology": {
                "logits": ideology_logits,
                "probabilities": ideology_probs,
                "predictions": torch.argmax(ideology_probs, dim=-1),
            },

            "propaganda": {
                "logits": propaganda_logits,
                "probabilities": propaganda_probs,
                "predictions": torch.argmax(propaganda_probs, dim=-1),
            },

            "narrative": narrative_outputs,

            "narrative_frame": narrative_frame_outputs,

            "emotion": emotion_outputs,
        }

        # ----------------------------------------------------
        # Loss
        # ----------------------------------------------------

        if labels is not None:

            loss_dict = {}

            if "bias" in labels:
                loss_dict["bias"] = self.loss_ce(
                    bias_logits, labels["bias"].long()
                ) * self.config.bias_weight

            if "ideology" in labels:
                loss_dict["ideology"] = self.loss_ce(
                    ideology_logits, labels["ideology"].long()
                ) * self.config.ideology_weight

            if "propaganda" in labels:
                loss_dict["propaganda"] = self.loss_ce(
                    propaganda_logits, labels["propaganda"].long()
                ) * self.config.propaganda_weight

            if "narrative" in labels:
                loss_dict["narrative"] = self.loss_bce(
                    narrative_outputs["logits"],
                    labels["narrative"].float(),
                ) * self.config.narrative_weight

            if "narrative_frame" in labels:
                loss_dict["frame"] = self.loss_bce(
                    narrative_frame_outputs["logits"],
                    labels["narrative_frame"].float(),
                ) * self.config.narrative_frame_weight

            if "emotion" in labels:
                loss_dict["emotion"] = self.loss_bce(
                    emotion_outputs["logits"],
                    labels["emotion"].float(),
                ) * self.config.emotion_weight

            if loss_dict:
                outputs["loss"] = torch.stack(list(loss_dict.values())).mean()
                outputs["loss_breakdown"] = loss_dict

        return outputs