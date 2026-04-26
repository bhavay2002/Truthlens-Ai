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
from ....training.loss_functions import LossConfig, LossFactory
from ....training.trainer import Trainer, TrainerConfig

logger = logging.getLogger(__name__)


@dataclass
class IdeologyClassifierConfig:
    model_name: str = "roberta-base"
    pooling: str = "cls"
    dropout: float = 0.1
    label_smoothing: float = 0.1
    device: Optional[str] = None
    use_regression_head: bool = False
    regression_output_dim: int = 1
    regression_hidden_dim: Optional[int] = None
    regression_activation: str = "gelu"


class IdeologyClassifier(BaseModel):

    NUM_CLASSES = 3

    def __init__(self, config: IdeologyClassifierConfig) -> None:
        super().__init__()

        if not isinstance(config, IdeologyClassifierConfig):
            raise TypeError("config must be IdeologyClassifierConfig")

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

        self.classifier_head = ClassificationHead(
            ClassificationHeadConfig(
                input_dim=self.encoder.hidden_size,
                num_classes=self.NUM_CLASSES,
                dropout=config.dropout,
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

        # -------------------------------------------------
        # Loss
        # -------------------------------------------------

        self.loss_fn = LossFactory.create(
            LossConfig(
                loss_type="multi_class",
                label_smoothing=config.label_smoothing,
            )
        )

        # -------------------------------------------------
        # Calibration
        # -------------------------------------------------

        self.temperature = nn.Parameter(torch.ones(1))

        logger.info(
            "IdeologyClassifier initialized | model=%s | classes=%d",
            config.model_name,
            self.NUM_CLASSES,
        )

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

        head_output = self.classifier_head(pooled_output)

        logits = head_output["logits"]

        temperature = torch.clamp(self.temperature, 0.5, 5.0)
        logits = logits / temperature

        probs = F.softmax(logits, dim=-1)
        preds = probs.argmax(dim=-1)
        confidence = probs.max(dim=-1).values

        outputs: Dict[str, Any] = {
            "logits": logits,
            "probabilities": probs,
            "predictions": preds,
            "confidence": confidence,
            "entropy": head_output["entropy"],
            "embeddings": pooled_output,
        }

        if self.regression_head is not None:
            outputs["regression"] = self.regression_head(pooled_output)

        if labels is not None:

            if labels.dim() != 1:
                raise ValueError("labels must be 1D tensor")

            loss = self.loss_fn(logits, labels.long())
            outputs["loss"] = loss

        return outputs

    # -----------------------------------------------------

    @torch.inference_mode()
    def predict(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:

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

        return {
            "predictions": outputs["predictions"],
            "probabilities": outputs["probabilities"],
            "confidence": outputs["confidence"],
        }

    # -----------------------------------------------------

    def get_output_labels(self) -> Dict[int, str]:
        return {
            0: "left",
            1: "center",
            2: "right",
        }

    # -----------------------------------------------------

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

        cfg = IdeologyClassifierConfig(
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

        return cls(cfg)

    # -----------------------------------------------------

    @classmethod
    def from_model_config(
        cls,
        model_config: MultiTaskModelConfig,
    ) -> "IdeologyClassifier":

        task_cfg = model_config.tasks.get("ideology")

        if task_cfg is None:
            raise KeyError("Task 'ideology' not found")

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
                model_config.metadata.get("ideology_label_smoothing", 0.1)
            ),
        )

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