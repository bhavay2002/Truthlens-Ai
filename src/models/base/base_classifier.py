from __future__ import annotations

import logging
from abc import abstractmethod
from typing import Any, Dict, Optional

import torch
import torch.nn.functional as F
from torch import nn

from .base_model import BaseModel

logger = logging.getLogger(__name__)


class BaseClassifier(BaseModel):

    def __init__(
        self,
        num_classes: int,
        input_dim: int,
        dropout: float = 0.1,
        multi_label: bool = False,
        label_smoothing: float = 0.0,
    ) -> None:
        super().__init__()

        if num_classes <= 0:
            raise ValueError("num_classes must be positive")

        if input_dim <= 0:
            raise ValueError("input_dim must be positive")

        if not (0.0 <= dropout <= 1.0):
            raise ValueError("dropout must be in [0,1]")

        self.num_classes = num_classes
        self.input_dim = input_dim
        self.multi_label = multi_label
        self.label_smoothing = label_smoothing

        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(input_dim, num_classes)

        if multi_label:
            self.loss_fn: nn.Module = nn.BCEWithLogitsLoss()
        else:
            self.loss_fn = nn.CrossEntropyLoss(
                label_smoothing=label_smoothing
            )

    # =====================================================
    # ENCODE
    # =====================================================

    @abstractmethod
    def encode(self, *inputs: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        raise NotImplementedError

    # =====================================================
    # FORWARD
    # =====================================================

    def forward(
        self,
        *inputs: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        return_features: bool = False,
        **kwargs: Any,
    ) -> Dict[str, torch.Tensor]:

        features = self.encode(*inputs, **kwargs)

        if features.dim() != 2:
            raise ValueError(f"Expected 2D features, got {features.shape}")

        if features.size(1) != self.input_dim:
            raise ValueError(
                f"Feature dim mismatch: expected {self.input_dim}, got {features.size(1)}"
            )

        features = self.dropout(features)
        logits = self.classifier(features)

        # -------------------------------------------------
        # PROBS
        # -------------------------------------------------

        if self.multi_label:
            probs = torch.sigmoid(logits)
            preds = (probs > 0.5).long()
        else:
            probs = F.softmax(logits, dim=-1)
            preds = torch.argmax(probs, dim=-1)

        confidence = probs.max(dim=-1).values
        entropy = -torch.sum(probs * torch.log(probs + 1e-12), dim=-1)

        output: Dict[str, torch.Tensor] = {
            "logits": logits,
            "probabilities": probs,
            "predictions": preds,
            "confidence": confidence,
            "entropy": entropy,
        }

        # -------------------------------------------------
        # LOSS
        # -------------------------------------------------

        if labels is not None:
            loss = self.compute_loss(logits, labels)
            output["loss"] = loss

        if return_features:
            output["features"] = features

        return output

    # =====================================================
    # LOSS
    # =====================================================

    def compute_loss(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:

        if self.multi_label:
            labels = labels.float()
            return self.loss_fn(logits, labels)

        return self.loss_fn(logits, labels.long())

    # =====================================================
    # PREDICT
    # =====================================================

    @torch.inference_mode()
    def predict(
        self,
        *inputs: torch.Tensor,
        **kwargs: Any,
    ) -> Dict[str, torch.Tensor]:

        was_training = self.training
        self.eval()

        try:
            outputs = self.forward(*inputs, **kwargs)
        finally:
            if was_training:
                self.train()

        return {
            "predictions": outputs["predictions"],
            "probabilities": outputs["probabilities"],
            "confidence": outputs["confidence"],
        }

    # =====================================================
    # LOGITS
    # =====================================================

    @torch.inference_mode()
    def predict_logits(
        self,
        *inputs: torch.Tensor,
        **kwargs: Any,
    ) -> torch.Tensor:

        was_training = self.training
        self.eval()

        try:
            outputs = self.forward(*inputs, **kwargs)
        finally:
            if was_training:
                self.train()

        return outputs["logits"]

    # =====================================================
    # CONFIG
    # =====================================================

    def get_config(self) -> Dict[str, Any]:

        return {
            "num_classes": self.num_classes,
            "input_dim": self.input_dim,
            "multi_label": self.multi_label,
            "label_smoothing": self.label_smoothing,
        }