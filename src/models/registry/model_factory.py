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

from ..config.model_config import ModelConfigLoader, MultiTaskModelConfig

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

        if model_type not in ModelFactory.SUPPORTED_MODELS:
            raise ValueError(
                f"Unsupported model_type '{model_type}'. "
                f"Supported models: {ModelFactory.SUPPORTED_MODELS}"
            )

        logger.info("Creating model: %s", model_type)

        if model_type == "bias_classifier":
            cfg = BiasClassifierConfig(**config)
            return BiasClassifier(cfg)

        if model_type == "ideology_classifier":
            cfg = IdeologyClassifierConfig(**config)
            return IdeologyClassifier(cfg)

        if model_type == "propaganda_detector":
            cfg = PropagandaDetectorConfig(**config)
            return PropagandaDetector(cfg)

        if model_type == "narrative_detector":
            cfg = NarrativeDetectorConfig(**config)
            return NarrativeDetector(cfg)

        if model_type == "emotion_classifier":
            cfg = EmotionClassifierConfig(**config)
            return EmotionClassifier(cfg)

        if model_type == "multitask_truthlens":
            cfg = MultiTaskTruthLensConfig(**config)
            return MultiTaskTruthLensModel(cfg)

        raise RuntimeError("Model creation failed unexpectedly")

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
