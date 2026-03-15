"""
File: truthlens_model.py

Purpose
-------
Defines the multimodal TruthLens AI architecture that integrates:

1. RoBERTa language representations
2. Bias detection signals
3. Emotion manipulation signals

This hybrid architecture improves misinformation detection by combining
deep contextual embeddings with symbolic feature signals.

Input
-----
input_ids : torch.Tensor
attention_mask : torch.Tensor
bias_score : torch.Tensor
emotion_score : torch.Tensor

Output
------
logits : torch.Tensor
"""

import logging
from typing import Optional

import torch
import torch.nn as nn
from transformers import RobertaModel

from src.utils.settings import load_settings

# ---------------------------------------------------------
# Logging
# ---------------------------------------------------------

logger = logging.getLogger(__name__)


# ---------------------------------------------------------
# Settings
# ---------------------------------------------------------

SETTINGS = load_settings()

MODEL_NAME = SETTINGS.model.name


# ---------------------------------------------------------
# TruthLens Multimodal Model
# ---------------------------------------------------------


class TruthLensModel(nn.Module):
    """
    Multimodal fake news detection model.

    Combines:
    - RoBERTa text embeddings
    - Bias detection score
    - Emotion manipulation score
    """

    def __init__(
        self,
        roberta_model_name: Optional[str] = MODEL_NAME,
        hidden_size: int = 768,
        num_labels: int = 2,
    ):
        super().__init__()

        # -------------------------------------------------
        # RoBERTa Encoder
        # -------------------------------------------------

        self.roberta = RobertaModel.from_pretrained(roberta_model_name)

        # -------------------------------------------------
        # Feature Fusion Layer
        # -------------------------------------------------

        # 768 (CLS) + 2 additional signals
        fusion_input_size = hidden_size + 2

        self.classifier = nn.Sequential(
            nn.Linear(fusion_input_size, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, num_labels),
        )

        logger.info("TruthLens multimodal model initialized")

    # ---------------------------------------------------------
    # Forward Pass
    # ---------------------------------------------------------

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        bias_score: torch.Tensor,
        emotion_score: torch.Tensor,
    ):
        """
        Forward pass of the model.
        """

        # -------------------------------------------------
        # RoBERTa Forward
        # -------------------------------------------------

        outputs = self.roberta(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

        cls_embedding = outputs.last_hidden_state[:, 0, :]  # CLS token

        # -------------------------------------------------
        # Feature Fusion
        # -------------------------------------------------

        extra_features = torch.stack(
            [bias_score, emotion_score],
            dim=1,
        )

        fused_vector = torch.cat(
            [cls_embedding, extra_features],
            dim=1,
        )

        # -------------------------------------------------
        # Classification
        # -------------------------------------------------

        logits = self.classifier(fused_vector)

        return logits
