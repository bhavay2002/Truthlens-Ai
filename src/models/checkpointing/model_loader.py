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

        self.model_dir = Path(model_dir)

        if not self.model_dir.exists():
            raise FileNotFoundError(f"Model directory not found: {self.model_dir}")

        self.device = torch.device(
            device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        )

    # -------------------------------------------------
    # Public API
    # -------------------------------------------------

    def load(self) -> Dict[str, Any]:

        try:

            logger.info("Loading model from: %s", self.model_dir)

            tokenizer = self._load_tokenizer()
            model = self._load_model()

            model.to(self.device)
            model.eval()

            metadata = self._load_metadata_optional()

            logger.info("Model loaded successfully on device: %s", self.device)

            return {
                "model": model,
                "tokenizer": tokenizer,
                "device": self.device,
                "metadata": metadata,
            }

        except Exception as exc:

            logger.exception("Model loading failed")

            raise RuntimeError("Model loading failed") from exc

    # -------------------------------------------------
    # Tokenizer
    # -------------------------------------------------

    def _load_tokenizer(self):

        try:

            tokenizer_path = self.model_dir / "tokenizer"

            if tokenizer_path.exists():
                return AutoTokenizer.from_pretrained(tokenizer_path)

            return AutoTokenizer.from_pretrained(self.model_dir)

        except Exception as exc:

            logger.exception("Failed to load tokenizer")

            raise RuntimeError("Tokenizer loading failed") from exc

    # -------------------------------------------------
    # Model Loader
    # -------------------------------------------------

    def _load_model(self):

        config_file = self.model_dir / "model_config.json"
        checkpoint_file = self.model_dir / "model.pt"

        # -------------------------------------------------
        # HuggingFace fallback
        # -------------------------------------------------

        if not config_file.exists():

            logger.info("Loading HuggingFace model")

            model = AutoModelForSequenceClassification.from_pretrained(self.model_dir)

            return model

        # -------------------------------------------------
        # TruthLens model via ModelFactory
        # -------------------------------------------------

        try:

            from src.models.registry.model_factory import ModelFactory
            from src.models.checkpointing.checkpoint_manager import CheckpointManager

            with open(config_file, "r", encoding="utf-8") as f:
                config_data = json.load(f)

            model_type = config_data.get("model_type")

            if model_type is None:
                raise ValueError("model_config.json missing 'model_type'")

            model_params = config_data.get("model_params", {})

            model = ModelFactory.create(model_type, model_params)

            # -------------------------------------------------
            # Direct model.pt loading
            # -------------------------------------------------

            if checkpoint_file.exists():

                state_dict = torch.load(
                    checkpoint_file,
                    map_location=self.device,
                )

                model.load_state_dict(state_dict)

                return model

            # -------------------------------------------------
            # Checkpoint manager fallback
            # -------------------------------------------------

            checkpoint_dir = self.model_dir / "checkpoint_bundle"

            if checkpoint_dir.exists():

                checkpoint_manager = CheckpointManager(checkpoint_dir)

                latest_checkpoint = checkpoint_manager.get_latest_checkpoint()

                if latest_checkpoint:

                    checkpoint_data = checkpoint_manager.load_checkpoint(
                        latest_checkpoint
                    )

                    if "model_state_dict" not in checkpoint_data:
                        raise RuntimeError(
                            "Checkpoint missing 'model_state_dict'"
                        )

                    model.load_state_dict(checkpoint_data["model_state_dict"])

            return model

        except Exception as exc:

            logger.exception("Failed to construct model from configuration")

            raise RuntimeError("Model construction failed") from exc

    # -------------------------------------------------
    # Optional Metadata
    # -------------------------------------------------

    def _load_metadata_optional(self) -> Dict[str, Any]:

        checkpoint_bundle_dir = self.model_dir / "checkpoint_bundle"

        if not checkpoint_bundle_dir.exists():
            return {}

        try:

            from src.models.artifacts.artifact_manager import ArtifactManager

            artifact_manager = ArtifactManager(checkpoint_bundle_dir)

            return artifact_manager.load_metadata()

        except FileNotFoundError:

            return {}

        except Exception:

            logger.warning("Failed to load checkpoint metadata", exc_info=True)

            return {}