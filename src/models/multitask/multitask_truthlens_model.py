"""
File Name: multitask_truthlens_model.py
Module: models.multitask
Description:
    Defines the core multi-task neural architecture used in the TruthLens AI
    system. The model uses a shared transformer encoder and multiple task-
    specific heads for tasks including:

        - bias detection (binary)
        - ideology classification (left/center/right)
        - propaganda detection (binary)
        - narrative role detection (hero/villain/victim)
        - narrative frame detection (RE/HI/CO/MO/EC)
        - emotion classification (20-label multi-label)

    The architecture follows modern multi-task NLP research practices where
    a shared contextual encoder learns a universal representation while
    task-specific heads specialize for downstream objectives.

Dependencies:
    logging
    typing
    dataclasses
    torch
    torch.nn
    models.encoder.transformer_encoder
    models.heads.classification_head
    models.heads.multilabel_head
Inputs:
    input_ids: Tensor (batch_size, sequence_length)
    attention_mask: Tensor (batch_size, sequence_length)
    labels (optional): Dict[str, Tensor]
Outputs:
    Dictionary containing logits, probabilities, predictions, and optional loss
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, Optional, Any, List

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..base.multitask_base_model import MultiTaskBaseModel
from ..encoder.transformer_encoder import TransformerEncoder
from ..heads.classification_head import ClassificationHead, ClassificationHeadConfig
from ..heads.multilabel_head import MultiLabelHead, MultiLabelHeadConfig
from ..ensemble.ensemble_model import EnsembleConfig, EnsembleModel
from ..ensemble.stacking_ensemble import StackingEnsembleConfig, StackingEnsembleModel
from ..ensemble.weighted_ensemble import WeightedEnsembleConfig, WeightedEnsembleModel
from ..training.loss_functions import LossConfig, LossFactory
from ..training.trainer import Trainer, TrainerConfig

logger = logging.getLogger(__name__)


# ------------------------------------------------------------
# Configuration
# ------------------------------------------------------------

@dataclass
class MultiTaskTruthLensConfig:

    model_name: str = "roberta-base"
    pooling: str = "cls"
    dropout: float = 0.1
    device: Optional[str] = None

    bias_weight: float = 1.0
    ideology_weight: float = 1.0
    propaganda_weight: float = 1.0
    narrative_weight: float = 1.0
    narrative_frame_weight: float = 1.0
    emotion_weight: float = 1.0


# ------------------------------------------------------------
# Model
# ------------------------------------------------------------

class MultiTaskTruthLensModel(MultiTaskBaseModel):

    # Label definitions
    BIAS_LABELS = ["non_bias", "bias"]
    IDEOLOGY_LABELS = ["left", "center", "right"]
    PROPAGANDA_LABELS = ["non_propaganda", "propaganda"]

    NARRATIVE_LABELS = ["hero", "villain", "victim"]

    FRAME_LABELS = ["RE", "HI", "CO", "MO", "EC"]

    NUM_BIAS = len(BIAS_LABELS)
    NUM_IDEOLOGY = len(IDEOLOGY_LABELS)
    NUM_PROPAGANDA = len(PROPAGANDA_LABELS)
    NUM_NARRATIVE = len(NARRATIVE_LABELS)
    NUM_NARRATIVE_FRAMES = len(FRAME_LABELS)

    NUM_EMOTIONS = 20

    def __init__(self, config: MultiTaskTruthLensConfig):

        task_configs = {
            "bias": {"num_classes": self.NUM_BIAS, "type": "classification"},
            "ideology": {"num_classes": self.NUM_IDEOLOGY, "type": "classification"},
            "propaganda": {"num_classes": self.NUM_PROPAGANDA, "type": "classification"},
            "narrative": {"num_classes": self.NUM_NARRATIVE, "type": "multilabel"},
            "narrative_frame": {
                "num_classes": self.NUM_NARRATIVE_FRAMES,
                "type": "multilabel",
            },
            "emotion": {"num_classes": self.NUM_EMOTIONS, "type": "multilabel"},
        }

        super().__init__(task_configs=task_configs)

        self.config = config

        # ----------------------------------------------------
        # Shared Encoder
        # ----------------------------------------------------

        self.encoder = TransformerEncoder(
            model_name=config.model_name,
            pooling=config.pooling,
            device=config.device,
        )

        hidden = self.encoder.hidden_size

        # ----------------------------------------------------
        # Task Heads
        # ----------------------------------------------------

        self.bias_head = ClassificationHead(
            ClassificationHeadConfig(hidden, self.NUM_BIAS, dropout=config.dropout)
        )

        self.ideology_head = ClassificationHead(
            ClassificationHeadConfig(hidden, self.NUM_IDEOLOGY, dropout=config.dropout)
        )

        self.propaganda_head = ClassificationHead(
            ClassificationHeadConfig(hidden, self.NUM_PROPAGANDA, dropout=config.dropout)
        )

        self.narrative_head = MultiLabelHead(
            MultiLabelHeadConfig(hidden, self.NUM_NARRATIVE, dropout=config.dropout)
        )

        self.narrative_frame_head = MultiLabelHead(
            MultiLabelHeadConfig(hidden, self.NUM_NARRATIVE_FRAMES, dropout=config.dropout)
        )

        self.emotion_head = MultiLabelHead(
            MultiLabelHeadConfig(hidden, self.NUM_EMOTIONS, dropout=config.dropout)
        )

        # Optional ensemble wrappers for classification heads.
        self.bias_ensemble: Optional[nn.Module] = None
        self.ideology_ensemble: Optional[nn.Module] = None
        self.propaganda_ensemble: Optional[nn.Module] = None

        # ----------------------------------------------------
        # Loss functions
        # ----------------------------------------------------

        self.loss_ce = LossFactory.create(
            LossConfig(loss_type="multi_class", label_smoothing=0.1)
        )
        self.loss_bce = LossFactory.create(
            LossConfig(loss_type="multi_label")
        )

        # temperature scaling
        self.temperature = nn.Parameter(torch.ones(1))

        logger.info("MultiTaskTruthLensModel initialized")

    def encode(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        **_: Any,
    ) -> torch.Tensor:
        encoder_outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        return encoder_outputs["pooled_output"]

    def configure_task_ensembles(
        self,
        *,
        bias_models: Optional[List[nn.Module]] = None,
        ideology_models: Optional[List[nn.Module]] = None,
        propaganda_models: Optional[List[nn.Module]] = None,
        strategy: str = "average",
        weights: Optional[List[float]] = None,
        bias_meta_model: Optional[nn.Module] = None,
        ideology_meta_model: Optional[nn.Module] = None,
        propaganda_meta_model: Optional[nn.Module] = None,
    ) -> None:
        """
        Integrate ensemble modules for task-specific classification heads.
        """
        self.bias_ensemble = self._build_task_ensemble(
            base_models=bias_models,
            strategy=strategy,
            weights=weights,
            meta_model=bias_meta_model,
        )
        self.ideology_ensemble = self._build_task_ensemble(
            base_models=ideology_models,
            strategy=strategy,
            weights=weights,
            meta_model=ideology_meta_model,
        )
        self.propaganda_ensemble = self._build_task_ensemble(
            base_models=propaganda_models,
            strategy=strategy,
            weights=weights,
            meta_model=propaganda_meta_model,
        )

    def _build_task_ensemble(
        self,
        *,
        base_models: Optional[List[nn.Module]],
        strategy: str,
        weights: Optional[List[float]],
        meta_model: Optional[nn.Module],
    ) -> Optional[nn.Module]:
        if not base_models:
            return None

        device_str = self.config.device or ("cuda" if torch.cuda.is_available() else "cpu")

        if strategy == "weighted_average":
            return WeightedEnsembleModel(
                models=base_models,
                config=WeightedEnsembleConfig(
                    weights=weights,
                    device=device_str,
                ),
            )

        if strategy == "stacking":
            if meta_model is None:
                raise ValueError("meta_model must be provided for stacking strategy.")
            return StackingEnsembleModel(
                base_models=base_models,
                meta_model=meta_model,
                config=StackingEnsembleConfig(device=device_str),
            )

        return EnsembleModel(
            models=base_models,
            config=EnsembleConfig(
                strategy=strategy,
                weights=weights,
                device=device_str,
            ),
        )

    @staticmethod
    def _extract_ensemble_logits(outputs: Any) -> torch.Tensor:
        if isinstance(outputs, torch.Tensor):
            return outputs
        if isinstance(outputs, dict):
            for key in ("logits", "ensemble_logits"):
                value = outputs.get(key)
                if isinstance(value, torch.Tensor):
                    return value
            for value in outputs.values():
                if isinstance(value, torch.Tensor):
                    return value
        raise RuntimeError("Unsupported ensemble output format.")

    # ------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Dict[str, Any]:

        pooled = self.encode(input_ids=input_ids, attention_mask=attention_mask)

        # stabilize temperature
        temperature = torch.clamp(self.temperature, 0.5, 5.0)

        # ----------------------------------------------------
        # Task heads
        # ----------------------------------------------------

        if self.bias_ensemble is not None:
            bias_logits = self._extract_ensemble_logits(self.bias_ensemble(pooled))
        else:
            bias_logits = self.bias_head(pooled)
        bias_logits = bias_logits / temperature

        if self.ideology_ensemble is not None:
            ideology_logits = self._extract_ensemble_logits(self.ideology_ensemble(pooled))
        else:
            ideology_logits = self.ideology_head(pooled)
        ideology_logits = ideology_logits / temperature

        if self.propaganda_ensemble is not None:
            propaganda_logits = self._extract_ensemble_logits(
                self.propaganda_ensemble(pooled)
            )
        else:
            propaganda_logits = self.propaganda_head(pooled)
        propaganda_logits = propaganda_logits / temperature

        narrative_outputs = self.narrative_head(pooled)
        narrative_frame_outputs = self.narrative_frame_head(pooled)
        emotion_outputs = self.emotion_head(pooled)

        bias_probs = F.softmax(bias_logits, dim=-1)
        ideology_probs = F.softmax(ideology_logits, dim=-1)
        propaganda_probs = F.softmax(propaganda_logits, dim=-1)

        outputs: Dict[str, Any] = {

            "embeddings": pooled,

            "bias": {
                "logits": bias_logits,
                "probabilities": bias_probs,
                "predictions": torch.argmax(bias_probs, dim=-1),
            },

            "ideology": {
                "logits": ideology_logits,
                "probabilities": ideology_probs,
                "predictions": torch.argmax(ideology_probs, dim=-1),
            },

            "propaganda": {
                "logits": propaganda_logits,
                "probabilities": propaganda_probs,
                "predictions": torch.argmax(propaganda_probs, dim=-1),
            },

            "narrative": narrative_outputs,

            "narrative_frame": narrative_frame_outputs,

            "emotion": emotion_outputs,
        }

        # ----------------------------------------------------
        # Loss
        # ----------------------------------------------------

        if labels is not None:

            loss_dict = {}

            if "bias" in labels:
                loss_dict["bias"] = self.loss_ce(
                    bias_logits, labels["bias"].long()
                ) * self.config.bias_weight

            if "ideology" in labels:
                loss_dict["ideology"] = self.loss_ce(
                    ideology_logits, labels["ideology"].long()
                ) * self.config.ideology_weight

            if "propaganda" in labels:
                loss_dict["propaganda"] = self.loss_ce(
                    propaganda_logits, labels["propaganda"].long()
                ) * self.config.propaganda_weight

            if "narrative" in labels:
                loss_dict["narrative"] = self.loss_bce(
                    narrative_outputs["logits"],
                    labels["narrative"].float(),
                ) * self.config.narrative_weight

            if "narrative_frame" in labels:
                loss_dict["frame"] = self.loss_bce(
                    narrative_frame_outputs["logits"],
                    labels["narrative_frame"].float(),
                ) * self.config.narrative_frame_weight

            if "emotion" in labels:
                loss_dict["emotion"] = self.loss_bce(
                    emotion_outputs["logits"],
                    labels["emotion"].float(),
                ) * self.config.emotion_weight

            if loss_dict:
                outputs["loss"] = torch.stack(list(loss_dict.values())).mean()
                outputs["loss_breakdown"] = loss_dict

        return outputs

    # ------------------------------------------------------------
    # Trainer factory
    # ------------------------------------------------------------

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
