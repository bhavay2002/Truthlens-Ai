"""
File Name: model_factory.py
Module: models.factory
Description:
    Implements a centralized factory responsible for constructing TruthLens
    model instances. The factory enables clean model instantiation based on
    configuration parameters and ensures that model creation is consistent
    across training, evaluation, and inference pipelines.

    Supported model types include:
        • bias_classifier
        • ideology_classifier
        • propaganda_detector
        • narrative_detector
        • emotion_classifier
        • multitask_truthlens

    The factory follows dependency injection principles and ensures that
    model configuration is validated before instantiation.

Dependencies:
    logging
    typing
    dataclasses
    torch.nn
    models.tasks.bias_classifier
    models.tasks.ideology_classifier
    models.tasks.propaganda_detector
    models.tasks.narrative_detector
    models.tasks.emotion_classifier
    models.multitask.multitask_truthlens_model
Inputs:
    Model configuration dictionary
Outputs:
    Instantiated PyTorch model
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Any

import torch.nn as nn

from ..config import ModelConfigLoader, MultiTaskModelConfig
from ..encoder.encoder_config import EncoderConfig
from src.utils import ensure_non_empty_text

from ..tasks.bias.bias_classifier import BiasClassifier, BiasClassifierConfig
from ..tasks.ideology.ideology_classifier import (
    IdeologyClassifier,
    IdeologyClassifierConfig,
)
from ..tasks.propaganda.propaganda_detector import (
    PropagandaDetector,
    PropagandaDetectorConfig,
)
from ..tasks.narrative.narrative_detector import (
    NarrativeDetector,
    NarrativeDetectorConfig,
)
from ..tasks.emotion.emotion_classifier import (
    EmotionClassifier,
    EmotionClassifierConfig,
)
from ..multitask.multitask_truthlens_model import (
    MultiTaskTruthLensModel,
    MultiTaskTruthLensConfig,
)

logger = logging.getLogger(__name__)


class ModelFactory:
    """
    Factory for constructing TruthLens models.
    """

    SUPPORTED_MODELS = {
        "bias_classifier",
        "ideology_classifier",
        "propaganda_detector",
        "narrative_detector",
        "emotion_classifier",
        "multitask_truthlens",
    }

    @staticmethod
    def _resolve_encoder_fields(config: Dict[str, Any]) -> Dict[str, Any]:
        if not isinstance(config, dict):
            return {}

        raw_encoder_cfg = config.get("encoder_config")
        if isinstance(raw_encoder_cfg, EncoderConfig):
            return {
                "model_name": raw_encoder_cfg.model_name,
                "pooling": raw_encoder_cfg.pooling,
                "device": raw_encoder_cfg.device,
            }

        if isinstance(raw_encoder_cfg, dict):
            encoder_cfg = EncoderConfig.from_dict(raw_encoder_cfg)
            return {
                "model_name": encoder_cfg.model_name,
                "pooling": encoder_cfg.pooling,
                "device": encoder_cfg.device,
            }

        return {}

    @staticmethod
    def _resolve_regression_fields(config: Dict[str, Any]) -> Dict[str, Any]:
        if not isinstance(config, dict):
            return {}

        raw = config.get("regression_config")
        if not isinstance(raw, dict):
            return {}

        resolved: Dict[str, Any] = {}

        if "enabled" in raw:
            resolved["use_regression_head"] = bool(raw["enabled"])
        if "output_dim" in raw:
            resolved["regression_output_dim"] = int(raw["output_dim"])
        if "hidden_dim" in raw:
            resolved["regression_hidden_dim"] = (
                int(raw["hidden_dim"]) if raw["hidden_dim"] is not None else None
            )
        if "activation" in raw:
            resolved["regression_activation"] = str(raw["activation"])

        return resolved

    @staticmethod
    def create(model_type: str, config: Dict[str, Any]) -> nn.Module:
        """
        Create model instance.

        Args:
            model_type:
                Type of model to construct.
            config:
                Configuration dictionary.

        Returns:
            Instantiated PyTorch model.
        """

        normalized_model_type = ensure_non_empty_text(model_type, name="model_type")

        if normalized_model_type not in ModelFactory.SUPPORTED_MODELS:
            raise ValueError(
                f"Unsupported model_type '{normalized_model_type}'. "
                f"Supported models: {ModelFactory.SUPPORTED_MODELS}"
            )

        logger.info("Creating model: %s", normalized_model_type)

        import copy
        merged_config = copy.deepcopy(config)
        merged_config.update(ModelFactory._resolve_encoder_fields(config))
        merged_config.update(ModelFactory._resolve_regression_fields(config))

        if normalized_model_type == "bias_classifier":
            cfg = BiasClassifierConfig(**merged_config)
            return BiasClassifier(cfg)

        if normalized_model_type == "ideology_classifier":
            cfg = IdeologyClassifierConfig(**merged_config)
            return IdeologyClassifier(cfg)

        if normalized_model_type == "propaganda_detector":
            cfg = PropagandaDetectorConfig(**merged_config)
            return PropagandaDetector(cfg)

        if normalized_model_type == "narrative_detector":
            cfg = NarrativeDetectorConfig(**merged_config)
            return NarrativeDetector(cfg)

        if normalized_model_type == "emotion_classifier":
            cfg = EmotionClassifierConfig(**merged_config)
            return EmotionClassifier(cfg)

        if normalized_model_type == "multitask_truthlens":
            cfg = MultiTaskTruthLensConfig(**merged_config)
            return MultiTaskTruthLensModel(cfg)

        raise RuntimeError("Model creation failed unexpectedly")

    @staticmethod
    def create_wrapper(
        model_type: str,
        config: Dict[str, Any],
        *,
        device: str | None = None,
    ):
        from src.models.inference.model_wrapper import ModelWrapper

        model = ModelFactory.create(model_type, config)
        return ModelWrapper(model=model, device=device)

    @staticmethod
    def create_predictor(
        model_type: str,
        config: Dict[str, Any],
        *,
        device: str | None = None,
    ):
        from src.models.inference.predictor import Predictor

        model = ModelFactory.create(model_type, config)
        return Predictor(model=model, device=device)

    @staticmethod
    def create_from_model_config(
        model_config: MultiTaskModelConfig,
    ) -> nn.Module:
        """
        Build a ``MultiTaskTruthLensModel`` directly from a
        ``MultiTaskModelConfig``.

        Parameters
        ----------
        model_config:
            Structured configuration loaded via
            ``ModelConfigLoader.load_multitask_config()``.

        Returns
        -------
        nn.Module
            Instantiated ``MultiTaskTruthLensModel``.
        """
        logger.info(
            "ModelFactory.create_from_model_config | encoder=%s",
            model_config.encoder.model_name,
        )
        return MultiTaskTruthLensModel.from_model_config(model_config)

    @staticmethod
    def create_task_from_model_config(
        task_name: str,
        model_config: MultiTaskModelConfig,
    ) -> nn.Module:
        if task_name == "bias":
            return BiasClassifier.from_model_config(model_config)
        if task_name == "ideology":
            return IdeologyClassifier.from_model_config(model_config)
        if task_name == "propaganda":
            return PropagandaDetector.from_model_config(model_config)
        if task_name == "narrative":
            return NarrativeDetector.from_model_config(model_config)
        if task_name == "emotion":
            return EmotionClassifier.from_model_config(model_config)

        raise ValueError(f"Unsupported task_name '{task_name}' for task model creation")

    @staticmethod
    def create_from_yaml(yaml_path: "str | Path") -> nn.Module:
        """
        Load a ``MultiTaskModelConfig`` from a YAML file and instantiate the
        corresponding model.

        The loader expects a YAML file structured as::

            encoder:
              model_name: roberta-base
              pooling: cls
            tasks:
              bias:
                num_labels: 2
                task_type: multi_class
              ...
            dropout: 0.1

        Parameters
        ----------
        yaml_path:
            Path to the model YAML configuration file.

        Returns
        -------
        nn.Module
        """
        model_config = ModelConfigLoader.load_multitask_config(yaml_path)
        logger.info("ModelFactory.create_from_yaml | path=%s", yaml_path)
        return ModelFactory.create_from_model_config(model_config)

    @staticmethod
    def create_from_checkpoint(
        model_dir: str | Path,
        device: str | None = None,
    ) -> nn.Module:
        """
        Construct model from checkpoint bundle/configured artifacts.
        """
        from ..checkpointing.model_loader import ModelLoader

        loaded = ModelLoader(model_dir=model_dir, device=device).load()
        model = loaded.get("model")

        if not isinstance(model, nn.Module):
            raise RuntimeError("Loaded checkpoint did not provide a valid torch model")

        return model
