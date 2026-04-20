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

import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional

import torch
import joblib
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from src.utils.settings import load_settings
from src.models.registry.model_factory import ModelFactory
from src.models.metadata.model_metadata import ModelMetadata
from src.models.metadata.model_versioning import ModelVersionInfo, ModelVersionRegistry


logger = logging.getLogger(__name__)

# Backward-compatible aliases used by legacy tests/callers.
RobertaTokenizer = AutoTokenizer
RobertaForSequenceClassification = AutoModelForSequenceClassification

MULTITASK_MODEL_TYPE = "multitask_truthlens"


def _load_multitask_model(model_path: Path, device: torch.device):
    """Load a saved MultiTaskTruthLensModel from pytorch_model.bin.

    The encoder is initialised from its *config* only (no pretrained-weight
    download), because the weights we care about are all in pytorch_model.bin.
    This avoids a ~500 MB round-trip to HuggingFace Hub.
    """
    from src.models.multitask.multitask_truthlens_model import (
        MultiTaskTruthLensConfig,
        MultiTaskTruthLensModel,
    )

    config_path = model_path / "config.json"
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    base_model_name: str = cfg.get("model_name", "roberta-base")

    model_cfg = MultiTaskTruthLensConfig(
        model_name=base_model_name,
        dropout=cfg.get("dropout", 0.1),
        pooling=cfg.get("pooling", "cls"),
        init_from_config_only=True,
    )

    model = MultiTaskTruthLensModel(config=model_cfg)

    # ── Load trained weights ──────────────────────────────────────────────────
    weights_path = model_path / "pytorch_model.bin"
    if weights_path.exists():
        state_dict = torch.load(weights_path, map_location=device, weights_only=True)
        model.load_state_dict(state_dict)
        logger.info("Loaded MultiTaskTruthLensModel weights from %s", weights_path)
    else:
        logger.warning("No pytorch_model.bin found; using randomly initialised weights")

    return model


# ---------------------------------------------------------
# Settings
# ---------------------------------------------------------

_SETTINGS = None


def _get_settings():
    global _SETTINGS
    if _SETTINGS is None:
        _SETTINGS = load_settings()
    return _SETTINGS


# ---------------------------------------------------------
# Model Registry
# ---------------------------------------------------------


class ModelRegistry:
    """
    Centralized model loading interface for TruthLens models.
    """

    @staticmethod
    def load_model(
        model_name: Optional[str] = "truthlens_model",
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

            settings = _get_settings()
            model_dir = Path(settings.model.path)
            vectorizer_path = Path(settings.paths.tfidf_vectorizer_path)

            model_path = model_dir
            if model_name:
                named_model_path = model_dir / model_name
                if named_model_path.exists():
                    model_path = named_model_path
                elif not model_dir.exists():
                    raise FileNotFoundError(f"Model path not found: {named_model_path}")
                else:
                    raise FileNotFoundError(f"Model path not found: {named_model_path}")

            device_obj = torch.device(
                device if device else ("cuda" if torch.cuda.is_available() else "cpu")
            )

            # -------------------------------------------------
            # Load config.json (if available) to resolve base model name
            # -------------------------------------------------

            saved_config_path = model_path / "config.json"
            saved_cfg: dict = {}
            if saved_config_path.exists():
                with open(saved_config_path, "r", encoding="utf-8") as f:
                    saved_cfg = json.load(f)

            saved_model_type = saved_cfg.get("model_type")

            # Determine the base model name for tokenizer fallback:
            # 1) model_name field in config.json (most specific)
            # 2) encoder.name from settings
            # 3) settings.model.name
            _settings_encoder_name = getattr(
                getattr(settings.model, "encoder", None), "name", None
            )
            _base_model_name = (
                saved_cfg.get("model_name")
                or _settings_encoder_name
                or settings.model.name
            )

            # -------------------------------------------------
            # Load Tokenizer (with graceful fallback to base model)
            # -------------------------------------------------

            try:
                tokenizer = RobertaTokenizer.from_pretrained(str(model_path))
                logger.info("Tokenizer loaded from model path: %s", model_path)
            except Exception as tok_err:
                logger.warning(
                    "Failed to load tokenizer from %s (%s); falling back to '%s'",
                    model_path,
                    tok_err,
                    _base_model_name,
                )
                tokenizer = RobertaTokenizer.from_pretrained(_base_model_name)
                logger.info("Tokenizer loaded from base model: %s", _base_model_name)

            # -------------------------------------------------
            # Load Model
            # -------------------------------------------------

            if model_type is None and saved_model_type == MULTITASK_MODEL_TYPE:
                model = _load_multitask_model(model_path, device_obj)
            elif model_type is None:
                model = RobertaForSequenceClassification.from_pretrained(model_path)
            else:
                config_path = model_path / "model_config.json"

                if not config_path.exists():
                    raise FileNotFoundError(
                        f"Missing model_config.json required for factory models: {config_path}"
                    )

                with open(config_path, "r", encoding="utf-8") as f:
                    config_dict = json.load(f)

                model = ModelFactory.create(model_type, config_dict)

                checkpoint_path = model_path / "model.pt"

                if checkpoint_path.exists():
                    state_dict = torch.load(checkpoint_path, map_location=device_obj, weights_only=True)
                    missing, unexpected = model.load_state_dict(state_dict, strict=False)

                    if missing:
                        logger.warning("Missing keys: %s", missing)
                    if unexpected:
                        logger.warning("Unexpected keys: %s", unexpected)

            if hasattr(model, "to"):
                model.to(device_obj)
            if hasattr(model, "eval"):
                model.eval()

            # -------------------------------------------------
            # Load Optional TF-IDF Vectorizer
            # -------------------------------------------------

            vectorizer = None

            if vectorizer_path.exists():

                try:
                    vectorizer = joblib.load(vectorizer_path)
                    logger.info("TF-IDF vectorizer loaded")

                except Exception as exc:
                    logger.warning("Failed to load vectorizer: %s", exc)

            else:
                logger.debug("No TF-IDF vectorizer found")

            # -------------------------------------------------
            # Load Optional ModelMetadata
            # -------------------------------------------------

            metadata: Optional[ModelMetadata] = None
            metadata_path = model_path / "metadata.json"
            if metadata_path.exists():
                try:
                    metadata = ModelMetadata.load_json(metadata_path)
                    logger.info("ModelMetadata loaded from %s", metadata_path)
                except Exception as meta_exc:
                    logger.warning("Failed to load ModelMetadata: %s", meta_exc)

            logger.info("Model registry load complete")

            return {
                "model": model,
                "tokenizer": tokenizer,
                "vectorizer": vectorizer,
                "device": device_obj,
                "metadata": metadata,
            }

        except Exception as exc:

            logger.exception("Failed to load model from registry")
            raise RuntimeError("Model registry loading failed") from exc

    @staticmethod
    def list_versions(
        model_name: str,
        registry_dir: Optional[str] = None,
    ) -> list:
        """
        List all registered versions for a model name.

        Parameters
        ----------
        model_name : str
        registry_dir : str, optional
            Registry directory. Defaults to MODEL_DIR.

        Returns
        -------
        list[ModelVersionInfo]
        """

        target_dir = Path(registry_dir) if registry_dir else MODEL_DIR
        registry = ModelVersionRegistry(target_dir)
        return registry.list_versions(model_name)

    @staticmethod
    def get_latest_version(
        model_name: str,
        registry_dir: Optional[str] = None,
    ) -> Optional[ModelVersionInfo]:
        """
        Retrieve the latest registered version for a model.

        Parameters
        ----------
        model_name : str
        registry_dir : str, optional

        Returns
        -------
        Optional[ModelVersionInfo]
        """

        target_dir = Path(registry_dir) if registry_dir else MODEL_DIR
        registry = ModelVersionRegistry(target_dir)
        return registry.get_latest_version(model_name)

    @staticmethod
    def get_version(
        model_name: str,
        version: str,
        registry_dir: Optional[str] = None,
    ) -> Optional[ModelVersionInfo]:
        """
        Retrieve a specific version of a model.

        Parameters
        ----------
        model_name : str
        version : str
        registry_dir : str, optional

        Returns
        -------
        Optional[ModelVersionInfo]
        """

        target_dir = Path(registry_dir) if registry_dir else MODEL_DIR
        registry = ModelVersionRegistry(target_dir)
        return registry.get_version(model_name, version)


# ---------------------------------------------------------
# Convenience Helper
# ---------------------------------------------------------


def get_model() -> Dict[str, Any]:
    """
    Retrieve default TruthLens model from registry.
    """

    return ModelRegistry.load_model()
