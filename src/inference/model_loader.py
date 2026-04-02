"""
File Name: model_loader.py
Module: Model Loading and Artifact Management
Description:
    Provides a production-grade model loading utility responsible for loading
    trained ML models and associated artifacts such as tokenizers, feature
    scalers, feature selectors, and metadata schemas.

    The loader centralizes artifact management and ensures that inference
    pipelines remain clean and decoupled from model storage details.

    Designed for scalable ML systems supporting PyTorch and HuggingFace models.

Dependencies:
    logging
    pathlib
    typing
    dataclasses
    json
    pickle
    torch
    transformers
    joblib

Inputs:
    Model directory containing trained artifacts.

Outputs:
    Loaded models, tokenizers, preprocessing pipelines, and schemas.
"""

from __future__ import annotations

import json
import logging
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import joblib
import torch
from transformers import AutoTokenizer, AutoModel

logger = logging.getLogger(__name__)


@dataclass
class ModelArtifacts:
    """
    Container for loaded model artifacts.
    """
    bias_model: Optional[torch.nn.Module] = None
    ideology_model: Optional[torch.nn.Module] = None
    emotion_model: Optional[torch.nn.Module] = None
    tokenizer: Optional[Any] = None
    feature_scaler: Optional[Any] = None
    feature_selector: Optional[Any] = None
    feature_schema: Optional[Dict[str, Any]] = None


class ModelLoader:
    """
    Centralized model loader responsible for loading ML artifacts used in
    production inference systems.
    """

    def __init__(self, models_dir: str, device: str = "auto") -> None:
        self.models_dir = Path(models_dir)
        self.device = self._resolve_device(device)

        if not self.models_dir.exists():
            raise FileNotFoundError(f"Model directory not found: {self.models_dir}")

        logger.info("Initializing ModelLoader with models directory: %s", self.models_dir)

    def _resolve_device(self, device: str) -> torch.device:
        """
        Resolve the device configuration.
        """
        if device == "auto":
            resolved = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            resolved = torch.device(device)

        logger.info("ModelLoader using device: %s", resolved)
        return resolved

    def _load_torch_model(self, path: Path) -> Optional[torch.nn.Module]:
        """
        Load a PyTorch model.
        """
        if not path.exists():
            logger.warning("Model file not found: %s", path)
            return None

        try:
            model = torch.load(path, map_location=self.device)
            if isinstance(model, torch.nn.Module):
                model.to(self.device)
                model.eval()

            logger.info("Loaded PyTorch model from %s", path)
            return model

        except Exception as exc:
            logger.exception("Failed to load PyTorch model: %s", path)
            raise RuntimeError(f"Error loading model: {path}") from exc

    def _load_tokenizer(self, path: Path) -> Optional[Any]:
        """
        Load HuggingFace tokenizer.
        """
        if not path.exists():
            logger.warning("Tokenizer path not found: %s", path)
            return None

        try:
            tokenizer = AutoTokenizer.from_pretrained(path)
            logger.info("Loaded tokenizer from %s", path)
            return tokenizer
        except Exception as exc:
            logger.exception("Failed to load tokenizer")
            raise RuntimeError("Tokenizer loading failed") from exc

    def _load_pickle(self, path: Path) -> Optional[Any]:
        """
        Load pickle artifact.
        """
        if not path.exists():
            logger.warning("Artifact not found: %s", path)
            return None

        try:
            with open(path, "rb") as f:
                obj = pickle.load(f)

            logger.info("Loaded pickle artifact: %s", path)
            return obj

        except Exception as exc:
            logger.exception("Failed to load pickle artifact")
            raise RuntimeError(f"Error loading artifact: {path}") from exc

    def _load_joblib(self, path: Path) -> Optional[Any]:
        """
        Load joblib artifact.
        """
        if not path.exists():
            logger.warning("Artifact not found: %s", path)
            return None

        try:
            obj = joblib.load(path)
            logger.info("Loaded joblib artifact: %s", path)
            return obj

        except Exception as exc:
            logger.exception("Failed to load joblib artifact")
            raise RuntimeError(f"Error loading artifact: {path}") from exc

    def _load_json(self, path: Path) -> Optional[Dict[str, Any]]:
        """
        Load JSON schema file.
        """
        if not path.exists():
            logger.warning("Schema file not found: %s", path)
            return None

        try:
            with open(path, "r", encoding="utf-8") as f:
                schema = json.load(f)

            logger.info("Loaded schema file: %s", path)
            return schema

        except Exception as exc:
            logger.exception("Failed to load JSON schema")
            raise RuntimeError(f"Error loading schema: {path}") from exc

    def load_all(self) -> ModelArtifacts:
        """
        Load all models and preprocessing artifacts.
        """
        artifacts = ModelArtifacts()

        bias_model_path = self.models_dir / "bias_model.pt"
        ideology_model_path = self.models_dir / "ideology_model.pt"
        emotion_model_path = self.models_dir / "emotion_model.pt"

        tokenizer_path = self.models_dir / "roberta_model"

        scaler_path = self.models_dir / "feature_scaler.pkl"
        selector_path = self.models_dir / "feature_selector.pkl"
        schema_path = self.models_dir / "feature_schema.json"

        artifacts.bias_model = self._load_torch_model(bias_model_path)
        artifacts.ideology_model = self._load_torch_model(ideology_model_path)
        artifacts.emotion_model = self._load_torch_model(emotion_model_path)

        artifacts.tokenizer = self._load_tokenizer(tokenizer_path)

        artifacts.feature_scaler = self._load_joblib(scaler_path)
        artifacts.feature_selector = self._load_joblib(selector_path)

        artifacts.feature_schema = self._load_json(schema_path)

        logger.info("All model artifacts loaded successfully")

        return artifacts

    def load_model(self, name: str) -> Optional[torch.nn.Module]:
        """
        Load a specific model by name.
        """
        path = self.models_dir / f"{name}.pt"
        return self._load_torch_model(path)

    def load_tokenizer(self, name: str = "roberta_model") -> Optional[Any]:
        """
        Load tokenizer from model directory.
        """
        path = self.models_dir / name
        return self._load_tokenizer(path)

    def load_scaler(self) -> Optional[Any]:
        """
        Load feature scaler.
        """
        return self._load_joblib(self.models_dir / "feature_scaler.pkl")

    def load_selector(self) -> Optional[Any]:
        """
        Load feature selector.
        """
        return self._load_joblib(self.models_dir / "feature_selector.pkl")

    def load_schema(self) -> Optional[Dict[str, Any]]:
        """
        Load feature schema.
        """
        return self._load_json(self.models_dir / "feature_schema.json")