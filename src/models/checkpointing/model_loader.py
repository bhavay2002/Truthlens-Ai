"""
File Name: model_loader.py
Module: models.inference
Description:
    Provides a standardized model loading utility for the TruthLens AI system.
    The module is responsible for reconstructing trained models, loading
    checkpoints, restoring tokenizer artifacts, and preparing models for
    inference or evaluation.

    It supports both HuggingFace transformer models and internal TruthLens
    PyTorch architectures created through the ModelFactory. The loader also
    ensures correct device placement and evaluation mode initialization.
    
Dependencies:
    logging
    pathlib
    typing
    torch
    transformers
    src.models.factory.model_factory
Inputs:
    model_dir : Path
Outputs:
    dict containing model, tokenizer, and device
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from src.models.factory.model_factory import ModelFactory

logger = logging.getLogger(__name__)


class ModelLoader:
    """
    Utility class responsible for loading trained TruthLens models.
    """

    def __init__(
        self,
        model_dir: str | Path,
        device: Optional[str] = None,
    ) -> None:
        """
        Initialize model loader.

        Parameters
        ----------
        model_dir : str | Path
            Directory containing model artifacts.
        device : Optional[str]
            Device to load the model onto.
        """

        self.model_dir = Path(model_dir)

        if not self.model_dir.exists():
            raise FileNotFoundError(f"Model directory not found: {self.model_dir}")

        self.device = torch.device(
            device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        )

    def load(self) -> Dict[str, Any]:
        """
        Load model and tokenizer.

        Returns
        -------
        Dict[str, Any]
        """

        try:

            logger.info("Loading model from: %s", self.model_dir)

            tokenizer = self._load_tokenizer()

            model = self._load_model()

            model.to(self.device)
            model.eval()

            logger.info("Model loaded successfully on device: %s", self.device)

            return {
                "model": model,
                "tokenizer": tokenizer,
                "device": self.device,
            }

        except Exception as exc:
            logger.exception("Model loading failed")
            raise RuntimeError("Model loading failed") from exc

    # -------------------------------------------------
    # Internal Loaders
    # -------------------------------------------------

    def _load_tokenizer(self):
        """
        Load tokenizer from model directory.
        """

        try:
            tokenizer = AutoTokenizer.from_pretrained(self.model_dir)
            return tokenizer

        except Exception as exc:
            logger.exception("Failed to load tokenizer")
            raise RuntimeError("Tokenizer loading failed") from exc

    def _load_model(self):
        """
        Load model architecture and checkpoint.
        """

        config_file = self.model_dir / "model_config.json"
        checkpoint_file = self.model_dir / "model.pt"

        # -------------------------------------------------
        # HuggingFace model fallback
        # -------------------------------------------------

        if not config_file.exists():
            logger.info("Loading HuggingFace model from directory")

            return AutoModelForSequenceClassification.from_pretrained(self.model_dir)

        # -------------------------------------------------
        # TruthLens model loading via ModelFactory
        # -------------------------------------------------

        try:

            with open(config_file, "r", encoding="utf-8") as f:
                config_data = json.load(f)

            model_type = config_data.get("model_type")

            if model_type is None:
                raise ValueError("model_config.json missing 'model_type'")

            model_params = config_data.get("model_params", {})

            model = ModelFactory.create(model_type, model_params)

            if checkpoint_file.exists():
                state_dict = torch.load(checkpoint_file, map_location=self.device)
                model.load_state_dict(state_dict)

            return model

        except Exception as exc:
            logger.exception("Failed to construct model from configuration")
            raise RuntimeError("Model construction failed") from exc