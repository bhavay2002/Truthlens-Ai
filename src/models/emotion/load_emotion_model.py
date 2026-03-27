"""
File Name: load_emotion_model.py
Module: Model Utilities - Emotion Model Loader
Description:
    Provides utilities for loading pretrained emotion classification models
    used in the TruthLens AI system. The module handles model initialization,
    loading saved checkpoints, device placement, and preparing the model for
    inference or evaluation in production pipelines.

Dependencies:
    logging
    typing
    torch
    yaml
    emotion_classifier (local module)

Inputs:
    Model checkpoint path
    YAML configuration file

Outputs:
    Loaded emotion classification model ready for inference
"""

import logging
from typing import Dict, Any, Optional

import torch
import yaml

from src.features.emotion.emotion_classifier import EmotionClassifier


logger = logging.getLogger(__name__)


class EmotionModelLoader:
    """
    Handles loading and initialization of pretrained emotion models.
    """

    def __init__(
        self,
        config_path: str,
        checkpoint_path: str,
        device: Optional[str] = None,
    ) -> None:
        """Initialize model loader."""

        if not isinstance(config_path, str) or not config_path:
            raise ValueError("config_path must be a valid string")

        if not isinstance(checkpoint_path, str) or not checkpoint_path:
            raise ValueError("checkpoint_path must be a valid string")

        self.config = self._load_config(config_path)

        self.device = torch.device(
            device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        )

        self.model = EmotionClassifier(
            model_name=self.config["model"]["encoder_model"],
            num_emotions=self.config["model"]["num_labels"],
            dropout=self.config["model"].get("dropout", 0.1),
            device=str(self.device),
        )

        self._load_checkpoint(checkpoint_path)

        self.model.eval()

        logger.info("Emotion model loaded successfully")

    def get_model(self) -> EmotionClassifier:
        """Return loaded emotion model."""

        return self.model

    def _load_checkpoint(self, checkpoint_path: str) -> None:
        """Load model weights from checkpoint."""

        try:
            checkpoint = torch.load(checkpoint_path, map_location=self.device)

            if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
                self.model.load_state_dict(checkpoint["model_state_dict"])
            else:
                self.model.load_state_dict(checkpoint)

        except Exception as exc:
            logger.exception("Failed to load checkpoint")
            raise RuntimeError("Model checkpoint loading failed") from exc

    def _load_config(self, path: str) -> Dict[str, Any]:
        """Load YAML configuration."""

        try:
            with open(path, "r", encoding="utf-8") as f:
                config = yaml.safe_load(f)

            if not isinstance(config, dict):
                raise ValueError("Invalid config format")

            return config

        except Exception as exc:
            logger.exception("Failed to load config")
            raise RuntimeError("Configuration loading failed") from exc
