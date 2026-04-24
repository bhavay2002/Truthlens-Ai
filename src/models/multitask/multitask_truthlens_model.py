from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import Dict, Optional, Any, List
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..base.multitask_base_model import MultiTaskBaseModel
from ..encoder.encoder_config import EncoderConfig
from ..encoder.encoder_factory import EncoderFactory
from ..heads.classification_head import ClassificationHead, ClassificationHeadConfig
from ..heads.multilabel_head import MultiLabelHead, MultiLabelHeadConfig
from ..heads.regression_head import RegressionHead, RegressionHeadConfig
from .multitask_loss import MultiTaskLoss
from .multitask_output import MultiTaskOutput
from ..ensemble.ensemble_model import EnsembleConfig, EnsembleModel
from ..ensemble.stacking_ensemble import StackingEnsembleConfig, StackingEnsembleModel
from ..ensemble.weighted_ensemble import WeightedEnsembleConfig, WeightedEnsembleModel
from ..config import (
    EncoderConfig as ModelEncoderConfig,
    ModelConfigLoader,
    MultiTaskModelConfig,
)
from ..training.loss_functions import LossConfig, LossFactory
from ..training.trainer import Trainer, TrainerConfig



logger = logging.getLogger(__name__)

def configure_cuda_kernels() -> None:
    """Configure optional CUDA kernel/backends.

    This intentionally does NOT run at import time to avoid global process
    side-effects for callers that import this module.
    """

    if torch.cuda.is_available():
        try:
            # H100/Ada path: prefer FlashAttention kernels first, then memory-efficient
            # as fallback. Keep math kernel disabled by default for throughput.
            torch.backends.cuda.enable_flash_sdp(
                os.environ.get("TRUTHLENS_ENABLE_FLASH_SDP", "1") == "1"
            )
            torch.backends.cuda.enable_mem_efficient_sdp(
                os.environ.get("TRUTHLENS_ENABLE_MEM_EFFICIENT_SDP", "1") == "1"
            )
            torch.backends.cuda.enable_math_sdp(
                os.environ.get("TRUTHLENS_ENABLE_MATH_SDP", "0") == "1"
            )
        except Exception:
            logger.warning("Flash SDP not supported, falling back safely")

# ------------------------------------------------------------
# Configuration
# ------------------------------------------------------------

@dataclass
class MultiTaskTruthLensConfig:

    model_name: str = "roberta-base"
    pooling: str = "cls"
    dropout: float = 0.1
    device: Optional[str] = None
    gradient_checkpointing: bool = False
    init_from_config_only: bool = False

    bias_weight: float = 1.0
    ideology_weight: float = 1.0
    propaganda_weight: float = 1.0
    narrative_weight: float = 1.0
    narrative_frame_weight: float = 1.0
    emotion_weight: float = 1.0

    use_regression_head: bool = False
    regression_output_dim: int = 1
    regression_hidden_dim: Optional[int] = None
    regression_activation: str = "gelu"

    enabled_tasks: List[str] = field(
        default_factory=lambda: [
            "bias",
            "ideology",
            "propaganda",
            "narrative",
            "narrative_frame",
            "emotion",
        ]
    )


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

        self.encoder = EncoderFactory.create_transformer_encoder(
            EncoderConfig(
                model_name=config.model_name,
                pooling=config.pooling,
                device=config.device,
                gradient_checkpointing=config.gradient_checkpointing,
                init_from_config_only=config.init_from_config_only,
            )
        )

        if config.gradient_checkpointing and hasattr(
            self.encoder, "gradient_checkpointing_enable"
        ):
            self.encoder.gradient_checkpointing_enable()

        hidden = self.encoder.hidden_size

        # ----------------------------------------------------
        # Task Heads
        # ----------------------------------------------------

        enabled = set(getattr(config, "enabled_tasks", []) or [])

        def _on(name: str) -> bool:
            if enabled:
                return name in enabled
            return bool(getattr(config, f"use_{name}_head", True))

        self.bias_head = (
            ClassificationHead(
                ClassificationHeadConfig(hidden, self.NUM_BIAS, dropout=config.dropout)
            )
            if _on("bias")
            else None
        )

        self.ideology_head = (
            ClassificationHead(
                ClassificationHeadConfig(hidden, self.NUM_IDEOLOGY, dropout=config.dropout)
            )
            if _on("ideology")
            else None
        )

        self.propaganda_head = (
            ClassificationHead(
                ClassificationHeadConfig(hidden, self.NUM_PROPAGANDA, dropout=config.dropout)
            )
            if _on("propaganda")
            else None
        )

        self.narrative_head = (
            MultiLabelHead(
                MultiLabelHeadConfig(hidden, self.NUM_NARRATIVE, dropout=config.dropout)
            )
            if _on("narrative")
            else None
        )

        self.narrative_frame_head = (
            MultiLabelHead(
                MultiLabelHeadConfig(hidden, self.NUM_NARRATIVE_FRAMES, dropout=config.dropout)
            )
            if _on("narrative_frame")
            else None
        )

        self.emotion_head = (
            MultiLabelHead(
                MultiLabelHeadConfig(hidden, self.NUM_EMOTIONS, dropout=config.dropout)
            )
            if _on("emotion")
            else None
        )

        if not any(
            [
                self.bias_head,
                self.ideology_head,
                self.propaganda_head,
                self.narrative_head,
                self.narrative_frame_head,
                self.emotion_head,
            ]
        ):
            raise ValueError("At least one task head must be enabled")

        self.bias_regression_head: Optional[RegressionHead] = None
        self.ideology_regression_head: Optional[RegressionHead] = None
        self.propaganda_regression_head: Optional[RegressionHead] = None
        self.narrative_regression_head: Optional[RegressionHead] = None

        if config.use_regression_head:
            reg_cfg = RegressionHeadConfig(
                input_dim=hidden,
                output_dim=config.regression_output_dim,
                hidden_dim=config.regression_hidden_dim,
                dropout=config.dropout,
                activation=config.regression_activation,
            )
            self.bias_regression_head = RegressionHead(reg_cfg)
            self.ideology_regression_head = RegressionHead(reg_cfg)
            self.propaganda_regression_head = RegressionHead(reg_cfg)
            self.narrative_regression_head = RegressionHead(reg_cfg)

        # Optional ensemble wrappers for classification heads.
        self.bias_ensemble: Optional[nn.Module] = None
        self.ideology_ensemble: Optional[nn.Module] = None
        self.propaganda_ensemble: Optional[nn.Module] = None

        # ----------------------------------------------------
        # Loss functions
        # ----------------------------------------------------

        self.multitask_loss = MultiTaskLoss.from_task_settings(
            {
                "bias": {"task_type": "multi_class", "weight": config.bias_weight},
                "ideology": {
                    "task_type": "multi_class",
                    "weight": config.ideology_weight,
                },
                "propaganda": {
                    "task_type": "multi_class",
                    "weight": config.propaganda_weight,
                },
                "narrative": {
                    "task_type": "multi_label",
                    "weight": config.narrative_weight,
                },
                "narrative_frame": {
                    "task_type": "multi_label",
                    "weight": config.narrative_frame_weight,
                },
                "emotion": {
                    "task_type": "multi_label",
                    "weight": config.emotion_weight,
                },
            }
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
        return encoder_outputs["sequence_output"][:, 0, :]

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

        device = next(self.parameters()).device
        device_str = str(device)

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

    @staticmethod
    def _extract_logits_safe(outputs: Any) -> torch.Tensor:
        if isinstance(outputs, torch.Tensor):
            return outputs
        if isinstance(outputs, dict):
            logits = outputs.get("logits")
            if isinstance(logits, torch.Tensor):
                return logits
            raise RuntimeError("Missing logits in head output")
        raise RuntimeError("Invalid head output format")

    # ------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: Optional[Dict[str, torch.Tensor]] = None,
        task: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Forward pass for the multi-task model.

        Parameters
        ----------
        input_ids, attention_mask:
            Standard transformer inputs.
        labels:
            Optional dict mapping task name → label tensor. When provided,
            loss is computed inside the model.
        task:
            When set to a task name (e.g. ``"bias"``), only that head is
            executed and only that task's loss is computed.  This is the
            task-wise training path — one head per forward pass, shared
            encoder updated by a single clean gradient.

            When ``None`` (default, inference path), all heads run and
            the combined multi-task loss is computed if labels are given.
        """

        pooled = self.encode(input_ids=input_ids, attention_mask=attention_mask)

        # stabilize temperature
        temperature = torch.clamp(self.temperature.exp(), 0.5, 5.0)

        # ----------------------------------------------------
        # Task heads — skip heads not selected by `task`
        # ----------------------------------------------------
        _run_all = task is None

        bias_logits = None
        ideology_logits = None
        propaganda_logits = None
        narrative_outputs = None
        narrative_frame_outputs = None
        emotion_outputs = None

        bias_probs = None
        ideology_probs = None
        propaganda_probs = None
        bias_preds = None
        ideology_preds = None
        propaganda_preds = None

        if self.bias_head is not None and (_run_all or task == "bias"):
            if self.bias_ensemble is not None:
                bias_logits = self._extract_ensemble_logits(self.bias_ensemble(pooled))
            else:
                bias_logits = self._extract_logits_safe(self.bias_head(pooled))
            if not self.training:
                bias_logits = bias_logits / temperature

        if self.ideology_head is not None and (_run_all or task == "ideology"):
            if self.ideology_ensemble is not None:
                ideology_logits = self._extract_ensemble_logits(
                    self.ideology_ensemble(pooled)
                )
            else:
                ideology_logits = self._extract_logits_safe(self.ideology_head(pooled))
            if not self.training:
                ideology_logits = ideology_logits / temperature

        if self.propaganda_head is not None and (_run_all or task == "propaganda"):
            if self.propaganda_ensemble is not None:
                propaganda_logits = self._extract_ensemble_logits(
                    self.propaganda_ensemble(pooled)
                )
            else:
                propaganda_logits = self._extract_logits_safe(self.propaganda_head(pooled))
            if not self.training:
                propaganda_logits = propaganda_logits / temperature

        if self.narrative_head is not None and (_run_all or task == "narrative"):
            narrative_outputs = self.narrative_head(pooled)
            if not self.training:
                narrative_outputs["logits"] = narrative_outputs["logits"] / temperature

        if self.narrative_frame_head is not None and (_run_all or task == "narrative_frame"):
            narrative_frame_outputs = self.narrative_frame_head(pooled)
            if not self.training:
                narrative_frame_outputs["logits"] = (
                    narrative_frame_outputs["logits"] / temperature
                )

        if self.emotion_head is not None and (_run_all or task == "emotion"):
            emotion_outputs = self.emotion_head(pooled)
            if not self.training:
                emotion_outputs["logits"] = emotion_outputs["logits"] / temperature

        def _compute_probs_and_preds() -> None:
            nonlocal bias_probs, ideology_probs, propaganda_probs
            nonlocal bias_preds, ideology_preds, propaganda_preds

            if bias_logits is not None:
                bias_probs = F.softmax(bias_logits, dim=-1)
                bias_preds = bias_probs.argmax(dim=-1)

            if ideology_logits is not None:
                ideology_probs = F.softmax(ideology_logits, dim=-1)
                ideology_preds = ideology_probs.argmax(dim=-1)

            if propaganda_logits is not None:
                propaganda_probs = F.softmax(propaganda_logits, dim=-1)
                propaganda_preds = propaganda_probs.argmax(dim=-1)

            if narrative_outputs is not None:
                probs = torch.sigmoid(narrative_outputs["logits"])
                narrative_outputs["probabilities"] = probs
                narrative_outputs["predictions"] = (probs > 0.5).int()

            if narrative_frame_outputs is not None:
                probs = torch.sigmoid(narrative_frame_outputs["logits"])
                narrative_frame_outputs["probabilities"] = probs
                narrative_frame_outputs["predictions"] = (probs > 0.5).int()

            if emotion_outputs is not None:
                probs = torch.sigmoid(emotion_outputs["logits"])
                emotion_outputs["probabilities"] = probs
                emotion_outputs["predictions"] = (probs > 0.5).int()

        if not self.training:
            with torch.no_grad():
                _compute_probs_and_preds()
        else:
            _compute_probs_and_preds()
        
        outputs: Dict[str, Any] = {
            "embeddings": pooled,
        }

        if bias_logits is not None:
            outputs["bias"] = {
                "logits": bias_logits,
                "probabilities": bias_probs,
                "predictions": bias_preds,
            }

        if ideology_logits is not None:
            outputs["ideology"] = {
                "logits": ideology_logits,
                "probabilities": ideology_probs,
                "predictions": ideology_preds,
            }

        if propaganda_logits is not None:
            outputs["propaganda"] = {
                "logits": propaganda_logits,
                "probabilities": propaganda_probs,
                "predictions": propaganda_preds,
            }

        if narrative_outputs is not None:
            outputs["narrative"] = narrative_outputs

        if narrative_frame_outputs is not None:
            outputs["narrative_frame"] = narrative_frame_outputs

        if emotion_outputs is not None:
            outputs["emotion"] = emotion_outputs

        if self.bias_regression_head is not None and "bias" in outputs:
            outputs["bias"]["regression"] = self.bias_regression_head(pooled)
        if self.ideology_regression_head is not None and "ideology" in outputs:
            outputs["ideology"]["regression"] = self.ideology_regression_head(pooled)
        if self.propaganda_regression_head is not None and "propaganda" in outputs:
            outputs["propaganda"]["regression"] = self.propaganda_regression_head(pooled)
        if self.narrative_regression_head is not None and "narrative" in outputs:
            outputs["narrative"]["regression"] = self.narrative_regression_head(pooled)

        multitask_output = MultiTaskOutput()
        if bias_logits is not None:
            multitask_output.add_task_output(
                task_name="bias",
                logits=bias_logits,
                probabilities=bias_probs,
                predictions=bias_preds,
            )
        if ideology_logits is not None:
            multitask_output.add_task_output(
                task_name="ideology",
                logits=ideology_logits,
                probabilities=ideology_probs,
                predictions=ideology_preds,
            )
        if propaganda_logits is not None:
            multitask_output.add_task_output(
                task_name="propaganda",
                logits=propaganda_logits,
                probabilities=propaganda_probs,
                predictions=propaganda_preds,
            )
        if narrative_outputs is not None:
            multitask_output.add_task_output(
                task_name="narrative",
                logits=narrative_outputs["logits"],
                probabilities=narrative_outputs.get("probabilities"),
                predictions=narrative_outputs.get("predictions"),
            )
        if narrative_frame_outputs is not None:
            multitask_output.add_task_output(
                task_name="narrative_frame",
                logits=narrative_frame_outputs["logits"],
                probabilities=narrative_frame_outputs.get("probabilities"),
                predictions=narrative_frame_outputs.get("predictions"),
            )
        if emotion_outputs is not None:
            multitask_output.add_task_output(
                task_name="emotion",
                logits=emotion_outputs["logits"],
                probabilities=emotion_outputs.get("probabilities"),
                predictions=emotion_outputs.get("predictions"),
            )

        # ----------------------------------------------------
        # Loss — single-task fast path (training) / multi-task (inference)
        # ----------------------------------------------------

        if labels is not None:
            active_tasks = list(labels.keys())

            if self.training and task is not None:
                # Single-task training: route through the fast path.
                # Only one head was executed so logits_for_loss is trivial.
                logits_for_loss: Dict[str, torch.Tensor] = {}
                if task == "bias" and bias_logits is not None:
                    logits_for_loss["bias"] = bias_logits
                elif task == "ideology" and ideology_logits is not None:
                    logits_for_loss["ideology"] = ideology_logits
                elif task == "propaganda" and propaganda_logits is not None:
                    logits_for_loss["propaganda"] = propaganda_logits
                elif task == "narrative" and narrative_outputs is not None:
                    logits_for_loss["narrative"] = narrative_outputs["logits"]
                elif task == "narrative_frame" and narrative_frame_outputs is not None:
                    logits_for_loss["narrative_frame"] = narrative_frame_outputs["logits"]
                elif task == "emotion" and emotion_outputs is not None:
                    logits_for_loss["emotion"] = emotion_outputs["logits"]

                if logits_for_loss and task in labels:
                    available_labels = {task: labels[task].to(pooled.device)}
                    total_loss, task_losses = self.multitask_loss(
                        logits=logits_for_loss,
                        labels=available_labels,
                    )
                    if not torch.isfinite(total_loss):
                        raise RuntimeError(
                            f"NaN/inf loss detected for task '{task}'"
                        )
                    if total_loss.item() > 1e4:
                        raise RuntimeError(
                            f"Exploding loss detected for task '{task}': "
                            f"{total_loss.item():.2f}"
                        )
                    outputs["loss"] = total_loss
                    outputs["task_losses"] = task_losses

                    if self.training:
                        active_logits = logits_for_loss.get(task)
                        return {
                            "loss": total_loss,
                            "task_losses": task_losses,
                            "logits": active_logits,
                        }

            else:
                # Multi-task path: used during inference/validation where all
                # heads ran (task is None).
                logits_for_loss = {}
                if bias_logits is not None and "bias" in labels:
                    logits_for_loss["bias"] = bias_logits
                if ideology_logits is not None and "ideology" in labels:
                    logits_for_loss["ideology"] = ideology_logits
                if propaganda_logits is not None and "propaganda" in labels:
                    logits_for_loss["propaganda"] = propaganda_logits
                if narrative_outputs is not None and "narrative" in labels:
                    logits_for_loss["narrative"] = narrative_outputs["logits"]
                if narrative_frame_outputs is not None and "narrative_frame" in labels:
                    logits_for_loss["narrative_frame"] = narrative_frame_outputs["logits"]
                if emotion_outputs is not None and "emotion" in labels:
                    logits_for_loss["emotion"] = emotion_outputs["logits"]

                available_labels: Dict[str, torch.Tensor] = {
                    key: labels[key].to(pooled.device)
                    for key in logits_for_loss
                    if key in labels
                }

                if available_labels:
                    total_loss, task_losses = self.multitask_loss(
                        logits=logits_for_loss,
                        labels=available_labels,
                    )
                    multitask_output.loss = total_loss
                    multitask_output.task_losses = task_losses
                    outputs["loss"] = total_loss
                    outputs["task_losses"] = task_losses
                    outputs["loss_breakdown"] = task_losses

        outputs["multitask_output"] = multitask_output

        return outputs

    # ------------------------------------------------------------
    # Trainer factory
    # ------------------------------------------------------------

    @classmethod
    def from_model_config(
        cls,
        model_config: MultiTaskModelConfig,
    ) -> "MultiTaskTruthLensModel":
        """
        Build a ``MultiTaskTruthLensModel`` from a central
        ``MultiTaskModelConfig``.

        Task-level weight overrides are read from ``model_config.metadata``
        using keys of the form ``"<task_name>_weight"`` (e.g.
        ``{"bias_weight": 2.0}``).

        Parameters
        ----------
        model_config:
            Structured configuration loaded via
            ``ModelConfigLoader.load_multitask_config()``.

        Returns
        -------
        MultiTaskTruthLensModel
        """
        cfg = MultiTaskTruthLensConfig(
            model_name=model_config.encoder.model_name,
            pooling=model_config.encoder.pooling,
            dropout=model_config.dropout,
            device=model_config.encoder.device,
            gradient_checkpointing=getattr(
                model_config.encoder, "gradient_checkpointing", False
            ),
            use_regression_head=bool(
                model_config.metadata.get("use_regression_head", False)
            ),
            regression_output_dim=int(
                model_config.metadata.get("regression_output_dim", 1)
            ),
            regression_hidden_dim=(
                int(model_config.metadata["regression_hidden_dim"])
                if model_config.metadata.get("regression_hidden_dim") is not None
                else None
            ),
            regression_activation=str(
                model_config.metadata.get("regression_activation", "gelu")
            ),
        )
        for weight_field in (
            "bias_weight",
            "ideology_weight",
            "propaganda_weight",
            "narrative_weight",
            "narrative_frame_weight",
            "emotion_weight",
        ):
            if weight_field in model_config.metadata:
                setattr(cfg, weight_field, float(model_config.metadata[weight_field]))

        logger.info(
            "MultiTaskTruthLensModel.from_model_config | model=%s dropout=%.2f",
            cfg.model_name,
            cfg.dropout,
        )
        return cls(cfg)

    @classmethod
    def load_from_yaml(
        cls,
        yaml_path: str | Path,
    ) -> "MultiTaskTruthLensModel":
        """
        Load a ``MultiTaskModelConfig`` from a YAML file and instantiate the
        model.

        Parameters
        ----------
        yaml_path:
            Path to the model YAML configuration file.

        Returns
        -------
        MultiTaskTruthLensModel
        """
        model_config = ModelConfigLoader.load_multitask_config(yaml_path)
        logger.info("MultiTaskTruthLensModel.load_from_yaml | path=%s", yaml_path)
        return cls.from_model_config(model_config)

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
