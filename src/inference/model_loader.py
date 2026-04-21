from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import joblib
import torch
from transformers import AutoTokenizer

from src.models.config import ModelConfigLoader, MultiTaskModelConfig
from src.models.metadata.model_metadata import ModelMetadata
from src.models.inference.model_wrapper import ModelWrapper
from src.models.inference.predictor import Predictor
from src.models.registry.model_factory import ModelFactory

logger = logging.getLogger(__name__)


# =========================================================
# Artifact Container
# =========================================================

@dataclass
class ModelArtifacts:
    bias_model: Optional[torch.nn.Module] = None
    ideology_model: Optional[torch.nn.Module] = None
    emotion_model: Optional[torch.nn.Module] = None

    tokenizer: Optional[Any] = None

    feature_scaler: Optional[Any] = None
    feature_selector: Optional[Any] = None
    feature_schema: Optional[Dict[str, Any]] = None

    model_metadata: Optional[ModelMetadata] = None
    model_config: Optional[MultiTaskModelConfig] = None

    bias_predictor: Optional[Predictor] = None
    ideology_predictor: Optional[Predictor] = None
    emotion_predictor: Optional[Predictor] = None

    multitask_model: Optional[torch.nn.Module] = None
    multitask_predictor: Optional[Predictor] = None


# =========================================================
# Model Loader
# =========================================================

class ModelLoader:

    def __init__(self, models_dir: str, device: str = "auto") -> None:
        self.models_dir = Path(models_dir)
        self.device = self._resolve_device(device)

        if not self.models_dir.exists():
            raise FileNotFoundError(f"Model directory not found: {self.models_dir}")

        logger.info("ModelLoader initialized at %s", self.models_dir)

    # -------------------------------------------------
    # Device
    # -------------------------------------------------

    def _resolve_device(self, device: str) -> torch.device:
        if device == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(device)

    # -------------------------------------------------
    # Core Loaders
    # -------------------------------------------------

    def _load_torch_model(self, path: Path) -> Optional[torch.nn.Module]:

        if not path.exists():
            logger.warning("Model not found: %s", path)
            return None

        try:
            # CPU-first load with compatibility fallback across torch versions
            obj = torch.load(path, map_location="cpu")

            if isinstance(obj, dict) and "state_dict" in obj:
                raise RuntimeError(
                    f"{path} contains state_dict, not full model. Use ModelFactory to rebuild."
                )

            if not isinstance(obj, torch.nn.Module):
                raise RuntimeError(f"Unsupported model object at {path}: {type(obj)}")

            model = obj

            #  Half precision (GPU only)
            if self.device.type == "cuda":
                try:
                    model = model.to(dtype=torch.float16)
                except Exception:
                    logger.debug("FP16 conversion skipped")

            #  Efficient transfer
            model.to(self.device)

            #  torch.compile (safe fallback)
            if hasattr(torch, "compile") and self.device.type == "cuda":
                try:
                    model = torch.compile(model, mode="reduce-overhead")
                except Exception:
                    logger.debug("torch.compile skipped")

            model.eval()

            logger.info("Loaded model: %s", path)
            return model

        except Exception as exc:
            logger.exception("Failed loading model: %s", path)
            raise RuntimeError(f"Error loading model: {path}") from exc

    def _load_tokenizer(self, path: Path) -> Optional[Any]:

        if not path.exists():
            logger.warning("Tokenizer not found: %s", path)
            return None

        try:
            tokenizer = AutoTokenizer.from_pretrained(path, use_fast=True)
            return tokenizer
        except Exception as exc:
            logger.exception("Tokenizer load failed")
            raise RuntimeError("Tokenizer load failed") from exc

    def _load_joblib(self, path: Path) -> Optional[Any]:
        if not path.exists():
            return None
        return joblib.load(path)

    def _load_json(self, path: Path) -> Optional[Dict[str, Any]]:
        if not path.exists():
            return None
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    # -------------------------------------------------
    # Main Loader
    # -------------------------------------------------

    def load_all(
        self,
        load_bias: bool = True,
        load_ideology: bool = True,
        load_emotion: bool = True,
    ) -> ModelArtifacts:

        artifacts = ModelArtifacts()

        # ---------------- Models ----------------

        if load_bias:
            artifacts.bias_model = self._load_torch_model(
                self.models_dir / "bias_model.pt"
            )

        if load_ideology:
            artifacts.ideology_model = self._load_torch_model(
                self.models_dir / "ideology_model.pt"
            )

        if load_emotion:
            artifacts.emotion_model = self._load_torch_model(
                self.models_dir / "emotion_model.pt"
            )

        # ---------------- Tokenizer ----------------

        tokenizer_path = self.models_dir / "tokenizer"
        if not tokenizer_path.exists():
            tokenizer_path = self.models_dir

        artifacts.tokenizer = self._load_tokenizer(tokenizer_path)

        # ---------------- Feature Artifacts ----------------

        artifacts.feature_scaler = self._load_joblib(
            self.models_dir / "feature_scaler.pkl"
        )

        artifacts.feature_selector = self._load_joblib(
            self.models_dir / "feature_selector.pkl"
        )

        artifacts.feature_schema = self._load_json(
            self.models_dir / "feature_schema.json"
        )

        # ---------------- Metadata ----------------

        artifacts.model_metadata = self.load_model_metadata()
        artifacts.model_config = self.load_model_config()

        # ---------------- Predictors ----------------

        artifacts.bias_predictor = self._build_predictor(artifacts.bias_model)
        artifacts.ideology_predictor = self._build_predictor(artifacts.ideology_model)
        artifacts.emotion_predictor = self._build_predictor(artifacts.emotion_model)

        # ---------------- Multitask ----------------

        artifacts.multitask_model = self.load_multitask_model(
            artifacts.model_config
        )

        artifacts.multitask_predictor = self._build_predictor(
            artifacts.multitask_model
        )

        logger.info("All artifacts loaded successfully")

        return artifacts

    # -------------------------------------------------
    # Builders
    # -------------------------------------------------

    def _build_predictor(self, model: Optional[torch.nn.Module]) -> Optional[Predictor]:
        if model is None:
            return None
        return Predictor(model=model, device=self.device)

    # -------------------------------------------------
    # Multitask
    # -------------------------------------------------

    def load_multitask_model(
        self,
        model_config: Optional[MultiTaskModelConfig],
    ) -> Optional[torch.nn.Module]:

        if model_config is None:
            return None

        try:
            model = ModelFactory.create_from_model_config(model_config)

            weights_path = self.models_dir / "multitask_model.pt"
            if weights_path.exists():
                state = torch.load(weights_path, map_location="cpu")
                if isinstance(state, dict) and "state_dict" in state:
                    state = state["state_dict"]
                if not isinstance(state, dict):
                    raise RuntimeError(
                        f"Unsupported multitask checkpoint format at {weights_path}: {type(state)}"
                    )
                model.load_state_dict(state)

            if self.device.type == "cuda":
                try:
                    model = model.to(dtype=torch.float16)
                except Exception:
                    logger.debug("FP16 conversion skipped for multitask model")

            model.to(self.device)

            if hasattr(torch, "compile") and self.device.type == "cuda":
                try:
                    model = torch.compile(model, mode="reduce-overhead")
                except Exception:
                    logger.debug("torch.compile skipped for multitask model")

            model.eval()
            return model

        except Exception as exc:
            logger.warning("Multitask model build failed: %s", exc)
            return None

    # -------------------------------------------------
    # Metadata
    # -------------------------------------------------

    def load_model_metadata(self) -> Optional[ModelMetadata]:

        path = self.models_dir / "metadata.json"

        if not path.exists():
            return None

        try:
            return ModelMetadata.load_json(path)
        except Exception as exc:
            logger.warning("Failed to load metadata: %s", exc)
            return None

    def load_model_config(self) -> Optional[MultiTaskModelConfig]:

        for name in ["config.yaml", "model_config.yaml"]:
            path = self.models_dir / name
            if path.exists():
                try:
                    return ModelConfigLoader.load_multitask_config(path)
                except Exception as exc:
                    logger.warning("Failed to load config: %s", exc)
                    return None

        return None

    # -------------------------------------------------
    # ONNX Export
    # -------------------------------------------------

    def export_onnx(self, model_name: str, output_path: str):

        model = self._load_torch_model(self.models_dir / f"{model_name}.pt")

        if model is None:
            raise ValueError(f"Model not found: {model_name}")

        dummy = {
            "input_ids": torch.ones(1, 16, dtype=torch.long).to(self.device),
            "attention_mask": torch.ones(1, 16, dtype=torch.long).to(self.device),
        }

        class _OnnxWrapper(torch.nn.Module):
            def __init__(self, wrapped: torch.nn.Module) -> None:
                super().__init__()
                self.wrapped = wrapped

            def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> Any:
                outputs = self.wrapped(input_ids=input_ids, attention_mask=attention_mask)
                return outputs.logits if hasattr(outputs, "logits") else outputs

        torch.onnx.export(
            _OnnxWrapper(model),
            (dummy["input_ids"], dummy["attention_mask"]),
            output_path,
            input_names=["input_ids", "attention_mask"],
            output_names=["logits"],
            dynamic_axes={
                "input_ids": {0: "batch", 1: "sequence"},
                "attention_mask": {0: "batch", 1: "sequence"},
            },
            opset_version=17,
        )

        logger.info("ONNX model exported to %s", output_path)