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
from ....training.trainer import Trainer, TrainerConfig

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

        if not isinstance(config, NarrativeDetectorConfig):
            raise TypeError("config must be NarrativeDetectorConfig")

        self.config = config

        # -------------------------------------------------
        # Encoder
        # -------------------------------------------------

        self.encoder = EncoderFactory.create_transformer_encoder(
            EncoderConfig(
                model_name=config.model_name,
                pooling=config.pooling,
                device=config.device,
            )
        )

        if hasattr(self.encoder, "gradient_checkpointing_enable"):
            self.encoder.gradient_checkpointing_enable()

        # -------------------------------------------------
        # Head
        # -------------------------------------------------

        self.classifier_head = MultiLabelHead(
            MultiLabelHeadConfig(
                input_dim=self.encoder.hidden_size,
                num_labels=self.NUM_LABELS,
                dropout=config.dropout,
                threshold=config.threshold,
                return_features=False,
            )
        )

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

    # -----------------------------------------------------
    # FORWARD
    # -----------------------------------------------------

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

        if not pooled_output.is_contiguous():
            pooled_output = pooled_output.contiguous()

        head_outputs = self.classifier_head(
            pooled_output,
            labels=labels,
        )

        outputs: Dict[str, Any] = {
            "logits": head_outputs["logits"],
            "probabilities": head_outputs["probabilities"],
            "predictions": head_outputs["predictions"],
            "confidence": head_outputs["confidence"],
            "entropy": head_outputs["entropy"],
            "loss": head_outputs.get("loss"),
            "embeddings": pooled_output,
        }

        if self.regression_head is not None:
            outputs["regression"] = self.regression_head(pooled_output)

        return outputs

    # -----------------------------------------------------
    # PREDICT
    # -----------------------------------------------------

    @torch.inference_mode()
    def predict(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        threshold: Optional[float] = None,
    ) -> Dict[str, Any]:

        was_training = self.training
        self.eval()

        try:
            outputs = self.forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )
        finally:
            if was_training:
                self.train()

        probs = outputs["probabilities"]

        if threshold is not None:
            preds = (probs >= float(threshold)).long()
        else:
            preds = outputs["predictions"]

        return {
            "predictions": preds,
            "probabilities": probs,
            "confidence": outputs["confidence"],
            "labels": self.LABEL_MAPPING,
        }

    # -----------------------------------------------------
    # LABELS
    # -----------------------------------------------------

    def get_output_labels(self) -> Dict[int, str]:
        return self.LABEL_MAPPING

    def get_label_list(self) -> List[str]:
        return list(self.LABELS)

    # -----------------------------------------------------
    # FACTORIES
    # -----------------------------------------------------

    @classmethod
    def from_task_config(
        cls,
        task_config: TaskConfig,
        head_config: HeadConfig,
        model_name: str = "roberta-base",
        pooling: str = "cls",
        device: Optional[str] = None,
        threshold: float = 0.5,
    ) -> "NarrativeDetector":

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

        return cls(cfg)

    @classmethod
    def from_model_config(
        cls,
        model_config: MultiTaskModelConfig,
    ) -> "NarrativeDetector":

        task_cfg = model_config.tasks.get("narrative")

        if task_cfg is None:
            raise KeyError("Task 'narrative' not found")

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
            threshold=float(
                model_config.metadata.get("narrative_threshold", 0.5)
            ),
        )

    # -----------------------------------------------------
    # TRAINER
    # -----------------------------------------------------

    def create_trainer(
        self,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[Any] = None,
        config: Optional[TrainerConfig] = None,
    ) -> Trainer:

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