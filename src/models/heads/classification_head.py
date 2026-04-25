from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional, Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F


logger = logging.getLogger(__name__)


@dataclass
class ClassificationHeadConfig:
    input_dim: int
    num_classes: int
    hidden_dim: Optional[int] = None
    dropout: float = 0.1
    activation: str = "gelu"
    use_layernorm: bool = False
    return_features: bool = False


class ClassificationHead(nn.Module):

    SUPPORTED_ACTIVATIONS = {
        "relu": nn.ReLU,
        "gelu": nn.GELU,
        "tanh": nn.Tanh,
    }

    def __init__(self, config: ClassificationHeadConfig) -> None:
        super().__init__()

        if config.input_dim <= 0:
            raise ValueError("input_dim must be positive")

        if config.num_classes <= 0:
            raise ValueError("num_classes must be positive")

        if not (0.0 <= config.dropout <= 1.0):
            raise ValueError("dropout must be between 0 and 1")

        if config.activation not in self.SUPPORTED_ACTIVATIONS:
            raise ValueError(f"Unsupported activation: {config.activation}")

        self.config = config
        self.has_hidden_layer = bool(config.hidden_dim)

        activation_cls = self.SUPPORTED_ACTIVATIONS[config.activation]

        if config.use_layernorm:
            self.norm = nn.LayerNorm(config.input_dim)
        else:
            self.norm = None

        if self.has_hidden_layer:

            if config.hidden_dim <= 0:
                raise ValueError("hidden_dim must be positive")

            self.fc1 = nn.Linear(config.input_dim, config.hidden_dim)
            self.activation = activation_cls()
            self.dropout = nn.Dropout(config.dropout)

            if config.use_layernorm:
                self.norm_hidden = nn.LayerNorm(config.hidden_dim)
            else:
                self.norm_hidden = None

            self.fc2 = nn.Linear(config.hidden_dim, config.num_classes)

        else:

            self.dropout = nn.Dropout(config.dropout)
            self.fc = nn.Linear(config.input_dim, config.num_classes)

        self._init_weights()

        logger.info(
            "ClassificationHead initialized | input_dim=%d | num_classes=%d",
            config.input_dim,
            config.num_classes,
        )

    # =====================================================
    # INIT
    # =====================================================

    def _init_weights(self):

        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    # =====================================================
    # FORWARD
    # =====================================================

    def forward(self, features: torch.Tensor) -> Dict[str, Any]:

        if features is None:
            raise ValueError("features cannot be None")

        if features.dim() != 2:
            raise ValueError(f"Expected 2D tensor, got {features.shape}")

        if features.size(1) != self.config.input_dim:
            raise ValueError(
                f"Expected input_dim={self.config.input_dim}, got {features.size(1)}"
            )

        if not features.is_contiguous():
            features = features.contiguous()

        x = features

        if self.norm is not None:
            x = self.norm(x)

        if self.has_hidden_layer:

            x = self.fc1(x)
            x = self.activation(x)

            if self.norm_hidden is not None:
                x = self.norm_hidden(x)

            x = self.dropout(x)

            logits = self.fc2(x)

        else:

            x = self.dropout(x)
            logits = self.fc(x)

        probs = F.softmax(logits, dim=-1)
        confidence = torch.max(probs, dim=-1).values
        entropy = -torch.sum(probs * torch.log(probs + 1e-12), dim=-1)

        output = {
            "logits": logits,
            "probabilities": probs,
            "confidence": confidence,
            "entropy": entropy,
        }

        if self.config.return_features:
            output["features"] = x

        return output

    # =====================================================
    # UTILS
    # =====================================================

    def get_output_dim(self) -> int:
        return self.config.num_classes