"""Hybrid Transformer + engineered-feature fusion model.

Moved here from ``src/features/fusion/feature_scaling.py`` (audit task 1).
That file is for the per-feature numeric scaler — the model does not
belong in the features layer.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from transformers import AutoModel


class HybridTruthLensModel(nn.Module):
    """Hybrid Transformer + Engineered Feature Model.

    Multi-head architecture:
      * encoder      : a HuggingFace transformer (RoBERTa / XLM-R / ...)
      * feature_proj : projection of the engineered feature vector into
                       ``hidden_dim``
      * fusion       : concat(cls, projected_features) → hidden_dim
      * task heads   : bias / propaganda / ideology / frame / narrative /
                       emotion

    The engineered feature vector must be pre-scaled by
    :class:`src.features.fusion.feature_scaling.FeatureScalingPipeline`
    using a scaler fitted on the training set.
    """

    def __init__(
        self,
        model_name: str,
        feature_dim: int,
        hidden_dim: int = 256,
        dropout: float = 0.2,
    ):
        super().__init__()

        self.encoder = AutoModel.from_pretrained(model_name)
        encoder_dim = self.encoder.config.hidden_size

        self.feature_proj = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
        )

        self.fusion = nn.Sequential(
            nn.Linear(encoder_dim + hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        self.bias_head = nn.Linear(hidden_dim, 1)
        self.propaganda_head = nn.Linear(hidden_dim, 1)
        self.ideology_head = nn.Linear(hidden_dim, 3)
        self.frame_head = nn.Linear(hidden_dim, 5)
        self.narrative_head = nn.Linear(hidden_dim, 3)
        self.emotion_head = nn.Linear(hidden_dim, 20)

    def forward(self, input_ids, attention_mask, features):
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        cls = outputs.last_hidden_state[:, 0]

        feat = self.feature_proj(features)

        fused = torch.cat([cls, feat], dim=1)
        fused = self.fusion(fused)

        return {
            "bias": self.bias_head(fused),
            "propaganda": self.propaganda_head(fused),
            "ideology": self.ideology_head(fused),
            "frame": self.frame_head(fused),
            "narrative": self.narrative_head(fused),
            "emotion": self.emotion_head(fused),
        }
