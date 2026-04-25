from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from src.models.checkpointing.validator import validate_checkpoint

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
            raise FileNotFoundError(self.model_dir)

        self.device = torch.device(
            device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        )

        self.use_half = use_half
        self.compile_model = compile_model

    # =====================================================
    # PUBLIC
    # =====================================================

    def load(self) -> Dict[str, Any]:

        logger.info("Loading model: %s", self.model_dir)

        tokenizer = self._load_tokenizer()
        model = self._load_model()

        model.to(self.device)

        if self.use_half and self.device.type == "cuda":
            model = model.half()

        if self.compile_model:
            try:
                model = torch.compile(model, mode="max-autotune")
            except Exception:
                logger.warning("compile failed")

        model.eval()

        metadata = self._load_metadata()

        return {
            "model": model,
            "tokenizer": tokenizer,
            "device": self.device,
            "metadata": metadata,
        }

    # =====================================================
    # TOKENIZER
    # =====================================================

    def _load_tokenizer(self):

        tok_path = self.model_dir / "tokenizer"

        try:
            if tok_path.exists():
                return AutoTokenizer.from_pretrained(tok_path, use_fast=True)

            return AutoTokenizer.from_pretrained(self.model_dir, use_fast=True)

        except Exception as e:
            logger.exception("Tokenizer load failed")
            raise RuntimeError from e

    # =====================================================
    # MODEL
    # =====================================================

    def _load_model(self):

        config_file = self.model_dir / "model_config.json"
        checkpoint_file = self.model_dir / "model.pt"

        # -------------------------
        # HF MODEL
        # -------------------------

        if not config_file.exists():

            return AutoModelForSequenceClassification.from_pretrained(
                self.model_dir,
                torch_dtype=torch.float16 if self.use_half else None,
                low_cpu_mem_usage=True,
            )

        # -------------------------
        # CUSTOM MODEL
        # -------------------------

        try:
            from src.models.registry.model_factory import ModelFactory
            from src.models.checkpointing.checkpoint_manager import CheckpointManager

            with open(config_file, "r") as f:
                cfg = json.load(f)

            model_type = cfg["model_type"]
            params = cfg.get("model_params", {})

            model = ModelFactory.create(model_type, params)

            # -------------------------
            # DIRECT CHECKPOINT
            # -------------------------

            if checkpoint_file.exists():

                checkpoint = torch.load(
                    checkpoint_file,
                    map_location="cpu",
                )

                state = (
                    checkpoint.get("model_state_dict")
                    if isinstance(checkpoint, dict)
                    else checkpoint
                )

                validate_checkpoint(state)

                res = model.load_state_dict(state, strict=False)

                if res.missing_keys:
                    raise RuntimeError(res.missing_keys)
                if res.unexpected_keys:
                    raise RuntimeError(res.unexpected_keys)

                return model

            # -------------------------
            # BUNDLE
            # -------------------------

            bundle = self.model_dir / "checkpoint_bundle"

            if bundle.exists():

                manager = CheckpointManager(bundle)
                latest = manager.latest()

                if latest:
                    checkpoint = manager.load_checkpoint(latest)

                    state = checkpoint.get("model_state_dict") or checkpoint.get("model")

                    validate_checkpoint(state)

                    res = model.load_state_dict(state, strict=False)

                    if res.missing_keys:
                        raise RuntimeError(res.missing_keys)
                    if res.unexpected_keys:
                        raise RuntimeError(res.unexpected_keys)

            return model

        except Exception as e:
            logger.exception("Model load failed")
            raise RuntimeError from e

    # =====================================================
    # METADATA
    # =====================================================

    def _load_metadata(self) -> Dict[str, Any]:

        bundle = self.model_dir / "checkpoint_bundle"

        if not bundle.exists():
            return {}

        try:
            from src.models.checkpointing.artifact_manager import ArtifactManager

            manager = ArtifactManager(bundle)
            return manager.load_metadata()

        except Exception:
            return {}