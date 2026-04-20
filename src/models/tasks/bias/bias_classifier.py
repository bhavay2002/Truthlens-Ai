"""
File Name: bias_classifier.py
Module: models.tasks.bias
Description:
    Implements a binary bias classification model used in the TruthLens AI system.
    The classifier predicts whether an article or text segment exhibits bias or
    is non-biased. The model uses a transformer encoder backbone (e.g., RoBERTa,
    BERT) followed by a classification head.

    This module is designed for production-grade ML pipelines and supports
    deterministic execution, structured outputs, and optional loss computation.

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
from ...config import HeadConfig, TaskConfig, MultiTaskModelConfig
from ...encoder.encoder_config import EncoderConfig
from ...encoder.encoder_factory import EncoderFactory
from ...heads.classification_head import (
    ClassificationHead,
    ClassificationHeadConfig,
)
from ...heads.regression_head import RegressionHead, RegressionHeadConfig
from ...training.loss_functions import LossConfig, LossFactory
from ...training.trainer import Trainer, TrainerConfig

logger = logging.getLogger(__name__)


@dataclass
class BiasClassifierConfig:
    """
    Configuration for the BiasClassifier.
    """

    model_name: str = "roberta-base"
    pooling: str = "cls"
    dropout: float = 0.1
    label_smoothing: float = 0.1
    device: Optional[str] = None
    use_regression_head: bool = False
    regression_output_dim: int = 1
    regression_hidden_dim: Optional[int] = None
    regression_activation: str = "gelu"


class BiasClassifier(BaseModel):
    """
    Binary bias classifier.

    Predicts:
        0 → non_bias
        1 → bias
    """

    NUM_CLASSES = 2

    def __init__(self, config: BiasClassifierConfig) -> None:
        super().__init__()

        if not isinstance(config, BiasClassifierConfig):
            raise TypeError("config must be BiasClassifierConfig")

        self.config = config

        # --------------------------------------------------
        # Transformer Encoder
        # --------------------------------------------------

        self.encoder = EncoderFactory.create_transformer_encoder(
            EncoderConfig(
                model_name=config.model_name,
                pooling=config.pooling,
                device=config.device,
            )
        )

        if hasattr(self.encoder, "gradient_checkpointing_enable"):
            self.encoder.gradient_checkpointing_enable()

        # --------------------------------------------------
        # Classification Head
        # --------------------------------------------------

        head_config = ClassificationHeadConfig(
            input_dim=self.encoder.hidden_size,
            num_classes=self.NUM_CLASSES,
            dropout=config.dropout,
        )

        self.classifier_head = ClassificationHead(head_config)

        self.regression_head: Optional[RegressionHead] = None
        if config.use_regression_head:
            self.regression_head = RegressionHead(
                RegressionHeadConfig(
                    input_dim=self.encoder.hidden_size,
                    output_dim=config.regression_output_dim,
                    hidden_dim=config.regression_hidden_dim,
                    dropout=config.dropout,
                    activation=config.regression_activation,
                )
            )

        # --------------------------------------------------
        # Loss
        # --------------------------------------------------

        self.loss_fn = LossFactory.create(
            LossConfig(
                loss_type="multi_class",
                label_smoothing=config.label_smoothing,
            )
        )

        # --------------------------------------------------
        # Temperature scaling for calibration
        # --------------------------------------------------

        self.temperature = nn.Parameter(torch.ones(1))

        logger.info(
            "BiasClassifier initialized | model=%s | hidden=%d",
            config.model_name,
            self.encoder.hidden_size,
        )

    # --------------------------------------------------

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
    ) -> Dict[str, Any]:

        if input_ids is None or attention_mask is None:
            raise ValueError("input_ids and attention_mask must be provided")

        encoder_outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

        pooled_output = encoder_outputs["pooled_output"]

        logits = self.classifier_head(pooled_output)

        # Temperature scaling
        temperature = torch.clamp(self.temperature, 0.5, 5.0)
        logits = logits / temperature

        if not self.training:
            probabilities = F.softmax(logits, dim=-1)
            predictions = probabilities.argmax(dim=-1)
            confidence = probabilities.max(dim=-1).values
        else:
            probabilities = None
            predictions = None
            confidence = None

        outputs: Dict[str, Any] = {
            "logits": logits,
            "probabilities": probabilities,
            "predictions": predictions,
            "confidence": confidence,
            "embeddings": pooled_output,
        }

        if self.regression_head is not None:
            outputs["regression"] = self.regression_head(pooled_output)

        if labels is not None:

            if labels.dim() != 1:
                raise ValueError("labels must be 1D tensor")

            if not ((labels >= 0).all() and (labels < self.NUM_CLASSES).all()):
                raise ValueError(
                    f"labels contain values outside [0, {self.NUM_CLASSES})"
                )

            loss = self.loss_fn(logits, labels.long())
            outputs["loss"] = loss

        return outputs

    # --------------------------------------------------

    @torch.inference_mode()
    def predict(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:

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

    # --------------------------------------------------

    def get_output_labels(self) -> Dict[int, str]:

        return {
            0: "non_bias",
            1: "bias",
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
    ) -> "BiasClassifier":
        """
        Instantiate a ``BiasClassifier`` from central config dataclasses.

        Parameters
        ----------
        task_config:
            Task-level descriptor from ``MultiTaskModelConfig.tasks["bias"]``.
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
        BiasClassifier
        """
        cfg = BiasClassifierConfig(
            model_name=model_name,
            pooling=pooling,
            dropout=head_config.dropout,
            label_smoothing=label_smoothing,
            device=device,
            use_regression_head=(
                task_config.regression.enabled
                if task_config.regression is not None
                else False
            ),
            regression_output_dim=(
                task_config.regression.output_dim
                if task_config.regression is not None
                else 1
            ),
            regression_hidden_dim=(
                task_config.regression.hidden_dim
                if task_config.regression is not None
                else None
            ),
            regression_activation=(
                task_config.regression.activation
                if task_config.regression is not None
                else "gelu"
            ),
        )
        logger.info(
            "BiasClassifier.from_task_config | task=%s num_labels=%d",
            task_config.name,
            task_config.num_labels,
        )
        return cls(cfg)

    @classmethod
    def from_model_config(
        cls,
        model_config: MultiTaskModelConfig,
    ) -> "BiasClassifier":
        task_cfg = model_config.tasks.get("bias")
        if task_cfg is None:
            raise KeyError("Task 'bias' not found in MultiTaskModelConfig")

        return cls.from_task_config(
            task_config=task_cfg,
            head_config=HeadConfig(
                input_dim=0,
                output_dim=task_cfg.num_labels,
                dropout=model_config.dropout,
            ),
            model_name=model_config.encoder.model_name,
            pooling=model_config.encoder.pooling,
            device=model_config.encoder.device,
            label_smoothing=float(
                model_config.metadata.get("bias_label_smoothing", 0.1)
            ),
        )

    def create_trainer(
        self,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[Any] = None,
        config: Optional[TrainerConfig] = None,
    ) -> Trainer:
        """
        Build a TruthLensTrainer for this model.

        Parameters
        ----------
        optimizer : torch.optim.Optimizer
        scheduler : optional LR scheduler
        config : TrainerConfig, optional
            Falls back to a default TrainerConfig if not supplied.

        Returns
        -------
        Trainer
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