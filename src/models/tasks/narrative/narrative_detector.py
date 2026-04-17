"""
File Name: narrative_detector.py
Module: models.tasks.narrative
Description:
    Implements a transformer-based narrative detection model for the TruthLens AI
    system. The model detects narrative roles and narrative frame signals within
    text using contextual embeddings produced by a pretrained transformer
    encoder followed by a multi-label classification head.

    The dataset contains the following narrative labels:

        hero
        villain
        victim
        hero_entities
        villain_entities
        victim_entities
        RE
        HI
        CO
        MO
        EC

    These labels represent narrative actors and narrative frame indicators.
    Because multiple narrative signals may appear simultaneously in a single
    article, the model performs multi-label classification using sigmoid
    activation and BCEWithLogitsLoss.

Dependencies:
    logging
    typing
    dataclasses
    torch
    torch.nn
    models.encoder.transformer_encoder
    models.heads.multilabel_head
Inputs:
    input_ids: Tensor (batch_size, sequence_length)
    attention_mask: Tensor (batch_size, sequence_length)
    labels (optional): Tensor (batch_size, 11)
Outputs:
    Dictionary containing logits, probabilities, predictions, and optional loss
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, Optional, List

import torch
import torch.nn as nn

from ...base.base_model import BaseModel
from ...config import HeadConfig, TaskConfig, MultiTaskModelConfig
from ...encoder.encoder_config import EncoderConfig
from ...encoder.encoder_factory import EncoderFactory
from ...heads.multilabel_head import MultiLabelHead, MultiLabelHeadConfig
from ...heads.regression_head import RegressionHead, RegressionHeadConfig
from ...training.trainer import Trainer, TrainerConfig

logger = logging.getLogger(__name__)


@dataclass
class NarrativeDetectorConfig:

    model_name: str = "roberta-base"
    pooling: str = "cls"
    dropout: float = 0.1
    threshold: float = 0.5
    device: Optional[str] = None
    use_regression_head: bool = False
    regression_output_dim: int = 1
    regression_hidden_dim: Optional[int] = None
    regression_activation: str = "gelu"


class NarrativeDetector(BaseModel):

    LABELS: List[str] = [
        "hero",
        "villain",
        "victim",
        "hero_entities",
        "villain_entities",
        "victim_entities",
        "RE",
        "HI",
        "CO",
        "MO",
        "EC",
    ]

    NUM_LABELS = len(LABELS)

    LABEL_MAPPING = {i: label for i, label in enumerate(LABELS)}

    def __init__(self, config: NarrativeDetectorConfig):

        super().__init__()

        self.config = config

        self.encoder = EncoderFactory.create_transformer_encoder(
            EncoderConfig(
                model_name=config.model_name,
                pooling=config.pooling,
                device=config.device,
            )
        )

        if hasattr(self.encoder, "gradient_checkpointing_enable"):
            self.encoder.gradient_checkpointing_enable()

        head_config = MultiLabelHeadConfig(
            input_dim=self.encoder.hidden_size,
            num_labels=self.NUM_LABELS,
            dropout=config.dropout,
            threshold=config.threshold,
        )

        self.classifier_head = MultiLabelHead(head_config)

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

        logger.info(
            "NarrativeDetector initialized | model=%s | labels=%d",
            config.model_name,
            self.NUM_LABELS,
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:

        encoder_outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

        pooled_output = encoder_outputs["pooled_output"]

        if not pooled_output.is_contiguous():
            pooled_output = pooled_output.contiguous()

        outputs = self.classifier_head(
            pooled_output,
            labels=labels,
        )

        if self.regression_head is not None:
            outputs["regression"] = self.regression_head(pooled_output)

        return outputs

    @torch.no_grad()
    def predict(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        threshold: Optional[float] = None,
    ) -> Dict[str, torch.Tensor]:

        was_training = self.training
        self.eval()

        outputs = self.forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

        if was_training:
            self.train()

        return {
            "predictions": outputs["predictions"],
            "probabilities": outputs["probabilities"],
            "labels": self.LABEL_MAPPING,
        }

    def get_output_labels(self) -> Dict[int, str]:

        return self.LABEL_MAPPING

    def get_label_list(self) -> List[str]:

                # thread-safe: avoid mutating config
                if threshold is not None:
                    outputs["probabilities"] = (
                        outputs["probabilities"] > float(threshold)
                    ).float()
    def from_task_config(
        cls,
        task_config: TaskConfig,
        head_config: HeadConfig,
        model_name: str = "roberta-base",
        pooling: str = "cls",
        device: Optional[str] = None,
        threshold: float = 0.5,
    ) -> "NarrativeDetector":
        """
        Instantiate a ``NarrativeDetector`` from central config dataclasses.

        Parameters
        ----------
        task_config:
            Task-level descriptor from
            ``MultiTaskModelConfig.tasks["narrative"]``.
        head_config:
            Head-level dimensions/dropout from ``model_config.HeadConfig``.
        model_name:
            HuggingFace model identifier.
        pooling:
            Encoder pooling strategy (``"cls"`` or ``"mean"``).
        device:
            Target device string or ``None`` for auto-detection.
        threshold:
            Sigmoid threshold for multi-label prediction (default ``0.5``).

        Returns
        -------
        NarrativeDetector
        """
        cfg = NarrativeDetectorConfig(
            model_name=model_name,
            pooling=pooling,
            dropout=head_config.dropout,
            threshold=threshold,
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
            "NarrativeDetector.from_task_config | task=%s num_labels=%d",
            task_config.name,
            task_config.num_labels,
        )
        return cls(cfg)

    @classmethod
    def from_model_config(
        cls,
        model_config: MultiTaskModelConfig,
    ) -> "NarrativeDetector":
        task_cfg = model_config.tasks.get("narrative")
        if task_cfg is None:
            raise KeyError("Task 'narrative' not found in MultiTaskModelConfig")

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
            threshold=float(model_config.metadata.get("narrative_threshold", 0.5)),
        )

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