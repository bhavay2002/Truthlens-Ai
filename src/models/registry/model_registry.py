"""
File Name: model_registry.py
Module: models.registry
Description:
    Centralized registry responsible for loading and managing models used
    in the TruthLens AI system. The registry standardizes how trained models,
    tokenizers, and supporting artifacts (such as TF-IDF vectorizers) are
    retrieved for inference, evaluation, and analysis.

    The module supports both HuggingFace-based models and internal TruthLens
    PyTorch models created via the ModelFactory. It ensures consistent loading
    behavior, artifact validation, and device placement.

Dependencies:
    logging
    typing
    pathlib
    torch
    joblib
    transformers
    src.utils.settings
    src.models.factory.model_factory
Inputs:
    model_name : str
Outputs:
    Dictionary containing loaded model, tokenizer, and optional vectorizer
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Any, Optional

import torch
import joblib
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from src.utils.settings import load_settings
from src.models.factory.model_factory import ModelFactory


logger = logging.getLogger(__name__)


# ---------------------------------------------------------
# Settings
# ---------------------------------------------------------

SETTINGS = load_settings()

MODEL_DIR = Path(SETTINGS.model.path)
VECTORIZER_PATH = Path(SETTINGS.paths.tfidf_vectorizer_path)


# ---------------------------------------------------------
# Model Registry
# ---------------------------------------------------------


class ModelRegistry:
    """
    Centralized model loading interface for TruthLens models.
    """

    @staticmethod
    def load_model(
        model_name: str = "truthlens_model",
        model_type: Optional[str] = None,
        device: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Load model and associated artifacts.

        Parameters
        ----------
        model_name : str
            Name of the model directory.
        model_type : Optional[str]
            Optional internal TruthLens model type.
        device : Optional[str]
            Target device.

        Returns
        -------
        Dict[str, Any]
        """

        try:

            logger.info("Loading model from registry: %s", model_name)

            model_path = MODEL_DIR / model_name if model_name else MODEL_DIR

            if not model_path.exists():
                raise FileNotFoundError(f"Model path not found: {model_path}")

            device_obj = torch.device(
                device if device else ("cuda" if torch.cuda.is_available() else "cpu")
            )

            # -------------------------------------------------
            # Load Tokenizer
            # -------------------------------------------------

            tokenizer = AutoTokenizer.from_pretrained(model_path)

            # -------------------------------------------------
            # Load Model
            # -------------------------------------------------

            if model_type is None:
                model = AutoModelForSequenceClassification.from_pretrained(model_path)
            else:
                config_path = model_path / "model_config.json"

                if not config_path.exists():
                    raise FileNotFoundError(
                        f"Missing model_config.json required for factory models: {config_path}"
                    )

                import json

                with open(config_path, "r", encoding="utf-8") as f:
                    config_dict = json.load(f)

                model = ModelFactory.create(model_type, config_dict)

                checkpoint_path = model_path / "model.pt"

                if checkpoint_path.exists():
                    state_dict = torch.load(checkpoint_path, map_location=device_obj)
                    model.load_state_dict(state_dict)

            model.to(device_obj)
            model.eval()

            # -------------------------------------------------
            # Load Optional TF-IDF Vectorizer
            # -------------------------------------------------

            vectorizer = None

            if VECTORIZER_PATH.exists():

                try:
                    vectorizer = joblib.load(VECTORIZER_PATH)
                    logger.info("TF-IDF vectorizer loaded")

                except Exception as exc:
                    logger.warning("Failed to load vectorizer: %s", exc)

            else:
                logger.debug("No TF-IDF vectorizer found")

            logger.info("Model registry load complete")

            return {
                "model": model,
                "tokenizer": tokenizer,
                "vectorizer": vectorizer,
                "device": device_obj,
            }

        except Exception as exc:

            logger.exception("Failed to load model from registry")
            raise RuntimeError("Model registry loading failed") from exc


# ---------------------------------------------------------
# Convenience Helper
# ---------------------------------------------------------


def get_model() -> Dict[str, Any]:
    """
    Retrieve default TruthLens model from registry.
    """

    return ModelRegistry.load_model()