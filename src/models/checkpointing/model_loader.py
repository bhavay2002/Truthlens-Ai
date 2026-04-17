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

    def __init__(
        self,
        model_dir: str | Path,
        device: Optional[str] = None,
        use_half: bool = True,
        compile_model: bool = False,
    ) -> None:

        self.model_dir = Path(model_dir)

        if not self.model_dir.exists():
            raise FileNotFoundError(f"Model directory not found: {self.model_dir}")

        self.device = torch.device(
            device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        )

        self.use_half = use_half
        self.compile_model = compile_model

    # -------------------------------------------------
    # Public API
    # -------------------------------------------------

    def load(self) -> Dict[str, Any]:

        logger.info("Loading model from: %s", self.model_dir)

        tokenizer = self._load_tokenizer()
        model = self._load_model()

        # Efficient device transfer
        model.to(self.device)

        # Optional inference optimization
        if self.use_half and self.device.type == "cuda":
            model = model.half()

        # torch.compile (PyTorch 2+)
        if self.compile_model:
            try:
                model = torch.compile(model, mode="max-autotune")
            except Exception:
                logger.warning("torch.compile failed, continuing without it")

        model.eval()

        metadata = self._load_metadata_optional()

        logger.info("Model ready on device: %s", self.device)

        return {
            "model": model,
            "tokenizer": tokenizer,
            "device": self.device,
            "metadata": metadata,
        }

    # -------------------------------------------------
    # Tokenizer (cached loading)
    # -------------------------------------------------

    def _load_tokenizer(self):

        tokenizer_path = self.model_dir / "tokenizer"

        try:
            if tokenizer_path.exists():
                return AutoTokenizer.from_pretrained(tokenizer_path, use_fast=True)

            return AutoTokenizer.from_pretrained(self.model_dir, use_fast=True)

        except Exception as exc:
            logger.exception("Tokenizer load failed")
            raise RuntimeError("Tokenizer loading failed") from exc

    # -------------------------------------------------
    # Model Loader
    # -------------------------------------------------

    def _load_model(self):

        config_file = self.model_dir / "model_config.json"
        checkpoint_file = self.model_dir / "model.pt"

        # -------------------------------------------------
        # HuggingFace direct load (fast path)
        # -------------------------------------------------

        if not config_file.exists():
            logger.info("Loading HuggingFace model")

            return AutoModelForSequenceClassification.from_pretrained(
                self.model_dir,
                torch_dtype=torch.float16 if self.use_half else None,
                low_cpu_mem_usage=True,
            )

        # -------------------------------------------------
        # TruthLens model
        # -------------------------------------------------

        try:
            from src.models.registry.model_factory import ModelFactory
            from src.models.checkpointing.checkpoint_manager import CheckpointManager

            with open(config_file, "r", encoding="utf-8") as f:
                config_data = json.load(f)

            model_type = config_data.get("model_type")
            model_params = config_data.get("model_params", {})

            if model_type is None:
                raise ValueError("model_config.json missing 'model_type'")

            model = ModelFactory.create(model_type, model_params)

            # -------------------------------------------------
            # Direct checkpoint (fastest)
            # -------------------------------------------------

            if checkpoint_file.exists():

                checkpoint = torch.load(
                    checkpoint_file,
                    map_location="cpu",  # avoid GPU spike
                    weights_only=True if hasattr(torch, "load") else False,
                )

                state_dict = (
                    checkpoint.get("model_state_dict")
                    if isinstance(checkpoint, dict)
                    else checkpoint
                )

                # Remove unnecessary keys
                if "_orig_mod" in str(type(model)):
                    model = model._orig_mod

                model.load_state_dict(state_dict, strict=False)

                return model

            # -------------------------------------------------
            # Checkpoint bundle fallback
            # -------------------------------------------------

            checkpoint_dir = self.model_dir / "checkpoint_bundle"

            if checkpoint_dir.exists():

                manager = CheckpointManager(checkpoint_dir)
                latest = manager.get_latest_checkpoint()

                if latest:
                    checkpoint = manager.load_checkpoint(latest)

                    state_dict = checkpoint.get("model")

                    if state_dict is None:
                        raise RuntimeError("Invalid checkpoint format")

                    model.load_state_dict(state_dict, strict=False)

            return model

        except Exception as exc:
            logger.exception("Model construction failed")
            raise RuntimeError("Model construction failed") from exc

    # -------------------------------------------------
    # Metadata
    # -------------------------------------------------

    def _load_metadata_optional(self) -> Dict[str, Any]:

        bundle_dir = self.model_dir / "checkpoint_bundle"

        if not bundle_dir.exists():
            return {}

        try:
            from src.models.checkpointing.artifact_manager import ArtifactManager

            manager = ArtifactManager(bundle_dir)
            return manager.load_metadata()

        except FileNotFoundError:
            return {}

        except Exception:
            logger.warning("Metadata load failed", exc_info=True)
            return {}