from __future__ import annotations

import torch
import torch.nn as nn
from transformers import AutoModel


class HybridTruthLensModel(nn.Module):
    """
    Hybrid Transformer + Engineered Feature Model
    """

    def __init__(
        self,
        model_name: str,
        feature_dim: int,
        hidden_dim: int = 256,
        dropout: float = 0.2,
    ):
        super().__init__()

        # -----------------------------
        # Transformer Encoder
        # -----------------------------
        self.encoder = AutoModel.from_pretrained(model_name)
        encoder_dim = self.encoder.config.hidden_size

        # -----------------------------
        # Feature Projection
        # -----------------------------
        self.feature_proj = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
        )

        # -----------------------------
        # Fusion Layer
        # -----------------------------
        self.fusion = nn.Sequential(
            nn.Linear(encoder_dim + hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # -----------------------------
        # Task Heads
        # -----------------------------

        # Binary heads
        self.bias_head = nn.Linear(hidden_dim, 1)
        self.propaganda_head = nn.Linear(hidden_dim, 1)

        # Multi-class
        self.ideology_head = nn.Linear(hidden_dim, 3)

        # Frame (multi-label)
        self.frame_head = nn.Linear(hidden_dim, 5)

        # Narrative roles
        self.narrative_head = nn.Linear(hidden_dim, 3)

        # Emotion (20 classes)
        self.emotion_head = nn.Linear(hidden_dim, 20)

    # -----------------------------------------------------

    def forward(
        self,
        input_ids,
        attention_mask,
        features,
    ):
        # -----------------------------
        # Transformer
        # -----------------------------
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

        cls = outputs.last_hidden_state[:, 0]  # CLS token

        # -----------------------------
        # Feature Path
        # -----------------------------
        feat = self.feature_proj(features)

        # -----------------------------
        # Fusion
        # -----------------------------
        fused = torch.cat([cls, feat], dim=1)
        fused = self.fusion(fused)

        # -----------------------------
        # Heads
        # -----------------------------
        return {
            "bias": self.bias_head(fused),
            "propaganda": self.propaganda_head(fused),
            "ideology": self.ideology_head(fused),
            "frame": self.frame_head(fused),
            "narrative": self.narrative_head(fused),
            "emotion": self.emotion_head(fused),
        }