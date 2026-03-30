"""
File Name: emotion_classifier.py
Module: Emotion Classification Model

Description:
    Transformer-based emotion classification model for the TruthLens AI system.

    Features:
        • configurable transformer backbone
        • multi-label or multi-class emotion prediction
        • flexible pooling strategies
        • optional encoder freezing
        • class weighting support
        • ML pipeline compatibility

"""

import logging
from typing import Optional, Dict

import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig


logger = logging.getLogger(__name__)


class EmotionClassifier(nn.Module):
    """
    Transformer-based emotion classification model.
    """

    def __init__(
        self,
        model_name: str,
        num_emotions: int,
        dropout: float = 0.1,
        multi_label: bool = False,
        freeze_encoder: bool = False,
        device: Optional[str] = None,
    ) -> None:

        super().__init__()

        if not isinstance(model_name, str) or not model_name.strip():
            raise ValueError("model_name must be a valid string")

        if not isinstance(num_emotions, int) or num_emotions <= 0:
            raise ValueError("num_emotions must be a positive integer")

        self.device = torch.device(
            device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        )

        self.multi_label = multi_label

        try:
            self.config = AutoConfig.from_pretrained(model_name)
            self.encoder = AutoModel.from_pretrained(model_name, config=self.config)
        except Exception as exc:
            logger.exception("Failed to load transformer model")
            raise RuntimeError("Transformer initialization failed") from exc

        hidden_size = self.config.hidden_size

        # Optional encoder freezing
        if freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False

        self.dropout = nn.Dropout(dropout)

        self.classifier = nn.Linear(hidden_size, num_emotions)

        if multi_label:
            self.activation = nn.Sigmoid()
        else:
            self.activation = nn.Softmax(dim=-1)

        self.to(self.device)

        logger.info(
            "EmotionClassifier initialized | model=%s | emotions=%d",
            model_name,
            num_emotions,
        )

    # -----------------------------------------------------

    def _pool_output(self, outputs) -> torch.Tensor:
        """
        Extract pooled embedding from transformer outputs.
        """

        if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
            return outputs.pooler_output

        # fallback to CLS token
        return outputs.last_hidden_state[:, 0]

    # -----------------------------------------------------

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        class_weights: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:

        if input_ids is None or attention_mask is None:
            raise ValueError("input tensors cannot be None")

        input_ids = input_ids.to(self.device)
        attention_mask = attention_mask.to(self.device)

        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

        pooled_output = self._pool_output(outputs)

        pooled_output = self.dropout(pooled_output)

        logits = self.classifier(pooled_output)

        probabilities = self.activation(logits)

        result: Dict[str, torch.Tensor] = {
            "logits": logits,
            "probabilities": probabilities,
        }

        # -------------------------------------------------
        # Loss computation
        # -------------------------------------------------

        if labels is not None:

            labels = labels.to(self.device)

            if self.multi_label:

                loss_fn = nn.BCEWithLogitsLoss(weight=class_weights)

            else:

                loss_fn = nn.CrossEntropyLoss(weight=class_weights)

                labels = labels.long()

            loss = loss_fn(logits, labels)

            result["loss"] = loss

        return result

    # -----------------------------------------------------

    def encode(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Extract transformer embeddings without classification head.
        Useful for feature analysis modules.
        """

        outputs = self.encoder(
            input_ids=input_ids.to(self.device),
            attention_mask=attention_mask.to(self.device),
        )

        pooled_output = self._pool_output(outputs)

        return pooled_output.detach()
