"""
File Name: ideology_classifier.py
Module: models.tasks.ideology
Description:
    Implements a transformer-based ideology classification model used in the
    TruthLens AI system. The classifier predicts ideological orientation from
    textual input using contextual embeddings from a pretrained transformer
    encoder and a task-specific classification head.

    The dataset contains three labels representing political ideology:
        0 -> left
        1 -> center
        2 -> right

    The model performs multi-class classification using CrossEntropyLoss.

Author: TruthLens Engineering
Date: 2026-04-02
Dependencies:
    logging
    typing
    dataclasses
    torch
    torch.nn
    models.encoder.transformer_encoder
    models.heads.classification_head
Inputs:
    input_ids: Tensor (batch_size, sequence_length)
    attention_mask: Tensor (batch_size, sequence_length)
    labels (optional): Tensor (batch_size)
Outputs:
    Dictionary containing logits, probabilities, predictions, and optional loss
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from ...base.base_model import BaseModel
from ...config.model_config import HeadConfig, TaskConfig
from ...encoder.transformer_encoder import TransformerEncoder
from ...heads.classification_head import (
    ClassificationHead,
    ClassificationHeadConfig,
)
from ...training.loss_functions import LossConfig, LossFactory
from ...training.trainer import Trainer, TrainerConfig

logger = logging.getLogger(__name__)


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

@dataclass
class IdeologyClassifierConfig:
    """
    Configuration for IdeologyClassifier.
    """

    model_name: str = "roberta-base"
    pooling: str = "cls"
    dropout: float = 0.1
    label_smoothing: float = 0.1
    device: Optional[str] = None


# ---------------------------------------------------------
# Ideology Classifier
# ---------------------------------------------------------

class IdeologyClassifier(BaseModel):
    """
    Transformer-based ideology classification model.

    Predicts:
        0 -> left
        1 -> center
        2 -> right
    """

    NUM_CLASSES = 3

    def __init__(self, config: IdeologyClassifierConfig) -> None:
        super().__init__()

        if not isinstance(config, IdeologyClassifierConfig):
            raise TypeError("config must be IdeologyClassifierConfig")

        self.config = config

        # -------------------------------------------------
        # Encoder
        # -------------------------------------------------

        self.encoder = TransformerEncoder(
            model_name=config.model_name,
            pooling=config.pooling,
            device=config.device,
        )

        # -------------------------------------------------
        # Classification Head
        # -------------------------------------------------

        head_config = ClassificationHeadConfig(
            input_dim=self.encoder.hidden_size,
            num_classes=self.NUM_CLASSES,
            dropout=config.dropout,
        )

        self.classifier_head = ClassificationHead(head_config)

        # -------------------------------------------------
        # Loss Function
        # -------------------------------------------------

        self.loss_fn = LossFactory.create(
            LossConfig(
                loss_type="multi_class",
                label_smoothing=config.label_smoothing,
            )
        )

        # -------------------------------------------------
        # Temperature Scaling (probability calibration)
        # -------------------------------------------------

        self.temperature = nn.Parameter(torch.ones(1))

        logger.info(
            "IdeologyClassifier initialized | model=%s | classes=%d",
            config.model_name,
            self.NUM_CLASSES,
        )

    # -----------------------------------------------------
    # Forward Pass
    # -----------------------------------------------------

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass.

        Args:
            input_ids:
                Token ids tensor (batch_size, seq_len)
            attention_mask:
                Attention mask tensor (batch_size, seq_len)
            labels:
                Optional ground truth labels

        Returns:
            Dictionary containing logits, probabilities, predictions,
            confidence scores, embeddings, and optional loss.
        """

        if input_ids is None or attention_mask is None:
            raise ValueError("input_ids and attention_mask must be provided")

        encoder_outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

        pooled_output = encoder_outputs["pooled_output"]

        logits = self.classifier_head(pooled_output)

        # Apply temperature scaling
        logits = logits / self.temperature

        probabilities = F.softmax(logits, dim=-1)

        predictions = torch.argmax(probabilities, dim=-1)

        confidence = torch.max(probabilities, dim=-1).values

        outputs: Dict[str, torch.Tensor] = {
            "logits": logits,
            "probabilities": probabilities,
            "predictions": predictions,
            "confidence": confidence,
            "embeddings": pooled_output,
        }

        # -------------------------------------------------
        # Loss computation (training)
        # -------------------------------------------------

        if labels is not None:

            if labels.dim() != 1:
                raise ValueError("labels must be 1D tensor (batch_size,)")

            loss = self.loss_fn(logits, labels.long())

            outputs["loss"] = loss

        return outputs

    # -----------------------------------------------------
    # Inference
    # -----------------------------------------------------

    @torch.inference_mode()
    def predict(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Run inference.

        Returns:
            predictions, probabilities, confidence
        """

        self.eval()

        outputs = self.forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

        return {
            "predictions": outputs["predictions"],
            "probabilities": outputs["probabilities"],
            "confidence": outputs["confidence"],
        }

    # -----------------------------------------------------
    # Label Mapping
    # -----------------------------------------------------

    def get_output_labels(self) -> Dict[int, str]:
        """
        Return ideology label mapping.
        """

        return {
            0: "left",
            1: "center",
            2: "right",
        }

    @classmethod
    def from_task_config(
        cls,
        task_config: TaskConfig,
        head_config: HeadConfig,
        model_name: str = "roberta-base",
        pooling: str = "cls",
        device: Optional[str] = None,
        label_smoothing: float = 0.1,
    ) -> "IdeologyClassifier":
        """
        Instantiate an ``IdeologyClassifier`` from central config dataclasses.

        Parameters
        ----------
        task_config:
            Task-level descriptor from ``MultiTaskModelConfig.tasks["ideology"]``.
        head_config:
            Head-level dimensions/dropout from ``model_config.HeadConfig``.
        model_name:
            HuggingFace model identifier.
        pooling:
            Encoder pooling strategy (``"cls"`` or ``"mean"``).
        device:
            Target device string or ``None`` for auto-detection.
        label_smoothing:
            Label smoothing factor for CrossEntropyLoss.

        Returns
        -------
        IdeologyClassifier
        """
        cfg = IdeologyClassifierConfig(
            model_name=model_name,
            pooling=pooling,
            dropout=head_config.dropout,
            label_smoothing=label_smoothing,
            device=device,
        )
        logger.info(
            "IdeologyClassifier.from_task_config | task=%s num_labels=%d",
            task_config.name,
            task_config.num_labels,
        )
        return cls(cfg)

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