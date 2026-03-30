"""
File Name: multitask_truthlens_model.py
Module: Model Architecture - Multi-Task TruthLens Model
Description:
    Defines the core multi-task neural architecture used in the TruthLens AI
    system. The model uses a shared transformer encoder and multiple task-
    specific classification heads for tasks such as bias detection, ideology
    classification, propaganda detection, narrative role detection, and
    emotion classification.

    The design follows modern multi-task NLP research practices where a shared
    language representation is learned by a common encoder while specialized
    heads handle individual tasks.

Dependencies:
    logging
    typing
    torch
    torch.nn
    transformer_encoder (local module)

Inputs:
    Tokenized transformer inputs (input_ids, attention_mask)
    Optional task labels

Outputs:
    Dictionary containing logits, probabilities, and optional losses
"""

import logging
from typing import Dict, Optional

import torch
import torch.nn as nn

from src.models.encoder.transformer_encoder import TransformerEncoder


logger = logging.getLogger(__name__)


class TaskHead(nn.Module):
    """
    Generic task-specific classification head.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        dropout: float = 0.1,
    ) -> None:
        """Initialize classification head."""

        super().__init__()

        if input_dim <= 0 or output_dim <= 0:
            raise ValueError("input_dim and output_dim must be positive")

        self.dropout = nn.Dropout(dropout)

        self.classifier = nn.Linear(input_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through classification head."""

        x = self.dropout(x)

        logits = self.classifier(x)

        return logits


class MultiTaskTruthLensModel(nn.Module):
    """
    Multi-task architecture combining shared encoder with multiple heads.

    Expected task schema:
    - bias: single-label (0/1)
    - ideology: single-label (0/1/2)
    - propaganda: single-label (0/1)
    - narrative: multi-label binary flags (hero/villain/victim)
    - emotion: single-label (0-19)
    """

    def __init__(
        self,
        encoder_model: str,
        num_bias_labels: int = 2,
        num_ideology_labels: int = 3,
        num_propaganda_labels: int = 2,
        num_narrative_labels: int = 3,
        num_emotion_labels: int = 20,
        dropout: float = 0.1,
        device: Optional[str] = None,
    ) -> None:
        """Initialize multi-task TruthLens model."""

        super().__init__()

        self.encoder = TransformerEncoder(
            model_name=encoder_model,
            pooling="cls",
            device=device,
        )

        hidden_size = self.encoder.hidden_size

        self.bias_head = TaskHead(hidden_size, num_bias_labels, dropout)

        self.ideology_head = TaskHead(hidden_size, num_ideology_labels, dropout)

        self.propaganda_head = TaskHead(hidden_size, num_propaganda_labels, dropout)

        self.narrative_head = TaskHead(hidden_size, num_narrative_labels, dropout)

        self.emotion_head = TaskHead(hidden_size, num_emotion_labels, dropout)

        self.num_bias_labels = num_bias_labels
        self.num_ideology_labels = num_ideology_labels
        self.num_propaganda_labels = num_propaganda_labels
        self.num_narrative_labels = num_narrative_labels
        self.num_emotion_labels = num_emotion_labels

        self.device = self.encoder.device

        self.to(self.device)

        logger.info("MultiTaskTruthLensModel initialized")

    @staticmethod
    def _prepare_single_label_targets(
        labels: torch.Tensor,
        *,
        num_classes: int,
        task_name: str,
    ) -> torch.Tensor:
        """
        Accept either class indices [batch] or one-hot [batch, num_classes].
        """

        if labels.dim() == 1:
            return labels.long()

        if labels.dim() == 2 and labels.size(1) == num_classes:
            return labels.argmax(dim=1).long()

        raise ValueError(
            f"{task_name} labels must be shape [batch] or [batch, {num_classes}]"
        )

    @staticmethod
    def _prepare_multi_label_targets(
        labels: torch.Tensor,
        *,
        num_classes: int,
        task_name: str,
    ) -> torch.Tensor:
        if labels.dim() == 1 and num_classes == 1:
            labels = labels.unsqueeze(1)

        if labels.dim() != 2 or labels.size(1) != num_classes:
            raise ValueError(
                f"{task_name} labels must be shape [batch, {num_classes}]"
            )

        return labels.float()

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Dict[str, torch.Tensor]:
        """Forward pass through encoder and all task heads."""

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

        narrative_logits = self.narrative_head(pooled)

        emotion_logits = self.emotion_head(pooled)

        outputs: Dict[str, torch.Tensor] = {
            "bias_logits": bias_logits,
            "ideology_logits": ideology_logits,
            "propaganda_logits": propaganda_logits,
            "narrative_logits": narrative_logits,
            "emotion_logits": emotion_logits,
            "bias_probabilities": torch.softmax(bias_logits, dim=1),
            "ideology_probabilities": torch.softmax(ideology_logits, dim=1),
            "propaganda_probabilities": torch.softmax(propaganda_logits, dim=1),
            "narrative_probabilities": torch.sigmoid(narrative_logits),
            "emotion_probabilities": torch.softmax(emotion_logits, dim=1),
        }

        if labels is not None:

            ce_loss = nn.CrossEntropyLoss()
            bce_loss = nn.BCEWithLogitsLoss()
            task_losses: Dict[str, torch.Tensor] = {}

            if "bias" in labels:
                bias_targets = self._prepare_single_label_targets(
                    labels["bias"].to(self.device),
                    num_classes=self.num_bias_labels,
                    task_name="bias",
                )
                task_losses["bias"] = ce_loss(
                    bias_logits,
                    bias_targets,
                )

            if "ideology" in labels:
                ideology_targets = self._prepare_single_label_targets(
                    labels["ideology"].to(self.device),
                    num_classes=self.num_ideology_labels,
                    task_name="ideology",
                )
                task_losses["ideology"] = ce_loss(
                    ideology_logits,
                    ideology_targets,
                )

            if "propaganda" in labels:
                propaganda_targets = self._prepare_single_label_targets(
                    labels["propaganda"].to(self.device),
                    num_classes=self.num_propaganda_labels,
                    task_name="propaganda",
                )
                task_losses["propaganda"] = ce_loss(
                    propaganda_logits,
                    propaganda_targets,
                )

            if "narrative" in labels:
                narrative_targets = self._prepare_multi_label_targets(
                    labels["narrative"].to(self.device),
                    num_classes=self.num_narrative_labels,
                    task_name="narrative",
                )
                task_losses["narrative"] = bce_loss(
                    narrative_logits,
                    narrative_targets,
                )

            if "emotion" in labels:
                emotion_targets = self._prepare_single_label_targets(
                    labels["emotion"].to(self.device),
                    num_classes=self.num_emotion_labels,
                    task_name="emotion",
                )
                task_losses["emotion"] = ce_loss(
                    emotion_logits,
                    emotion_targets,
                )

            if task_losses:
                total_loss = torch.stack(list(task_losses.values())).mean()
                outputs["loss"] = total_loss
                outputs["task_losses"] = task_losses

        return outputs
