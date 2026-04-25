from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, List

import joblib
import torch
import numpy as np
from transformers import AutoTokenizer

from src.models.config import ModelConfigLoader, MultiTaskModelConfig
from src.models.metadata.model_metadata import ModelMetadata
from src.models.inference.predictor import Predictor
from src.models.registry.model_factory import ModelFactory

logger = logging.getLogger(__name__)


# =========================================================
# ARTIFACT CONTAINER
# =========================================================

@dataclass
class ModelArtifacts:
    bias_model: Optional[torch.nn.Module] = None
    ideology_model: Optional[torch.nn.Module] = None
    emotion_model: Optional[torch.nn.Module] = None

    multitask_model: Optional[torch.nn.Module] = None

    tokenizer: Optional[Any] = None

    feature_scaler: Optional[Any] = None
    feature_selector: Optional[Any] = None
    feature_schema: Optional[Dict[str, Any]] = None

    model_metadata: Optional[ModelMetadata] = None
    model_config: Optional[MultiTaskModelConfig] = None

    bias_predictor: Optional[Predictor] = None
    ideology_predictor: Optional[Predictor] = None
    emotion_predictor: Optional[Predictor] = None
    multitask_predictor: Optional[Predictor] = None

    unified_predictor: Optional["UnifiedPredictor"] = None


# =========================================================
# UNIFIED PREDICTOR
# =========================================================

class UnifiedPredictor:

    def __init__(self, artifacts: ModelArtifacts, device: torch.device):
        self.artifacts = artifacts
        self.device = device

    def _format_output(self, raw: Dict[str, Any]):

        logits = raw.get("logits")
        probs = raw.get("probabilities")

        if logits is not None:
            logits = np.asarray(logits)

        if probs is None and logits is not None:
            probs = torch.softmax(torch.tensor(logits), dim=-1).numpy()

        preds = None
        if probs is not None:
            preds = np.argmax(probs, axis=1)

        return {
            "logits": logits,
            "probabilities": probs,
            "predictions": preds,
        }

    def predict_for_evaluation(self, texts: List[str]):

        # ---------------- MULTITASK ----------------
        if self.artifacts.multitask_predictor:
            raw = self.artifacts.multitask_predictor.predict(texts)
            return {"multitask": self._format_output(raw)}

        # ---------------- SINGLE TASK ----------------
        outputs = {}

        for name, predictor in {
            "bias": self.artifacts.bias_predictor,
            "ideology": self.artifacts.ideology_predictor,
            "emotion": self.artifacts.emotion_predictor,
        }.items():

            if predictor is None:
                continue

            raw = predictor.predict(texts)
            outputs[name] = self._format_output(raw)

        return outputs


# =========================================================
# MODEL LOADER
# =========================================================

class ModelLoader:

    def __init__(self, models_dir: str, device: str = "auto") -> None:
        self.models_dir = Path(models_dir)
        self.device = self._resolve_device(device)

        if not self.models_dir.exists():
            raise FileNotFoundError(f"Model directory not found: {self.models_dir}")

        logger.info("ModelLoader initialized at %s", self.models_dir)

    # =====================================================
    # DEVICE
    # =====================================================

    def _resolve_device(self, device: str) -> torch.device:
        if device == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(device)

    # =====================================================
    # LOAD HELPERS
    # =====================================================

    def _load_torch_model(self, path: Path) -> Optional[torch.nn.Module]:

        if not path.exists():
            logger.warning("Model not found: %s", path)
            return None

        obj = torch.load(path, map_location="cpu")

        if isinstance(obj, dict) and "state_dict" in obj:
            raise RuntimeError(f"{path} contains state_dict, not full model.")

        if not isinstance(obj, torch.nn.Module):
            raise RuntimeError(f"Unsupported model object: {type(obj)}")

        model = obj

        if self.device.type == "cuda":
            try:
                model = model.to(dtype=torch.float16)
            except Exception:
                pass

        model.to(self.device)

        if hasattr(torch, "compile") and self.device.type == "cuda":
            try:
                model = torch.compile(model)
            except Exception:
                pass

        model.eval()
        return model

    def _load_tokenizer(self, path: Path):

        if not path.exists():
            return None

        return AutoTokenizer.from_pretrained(path, use_fast=True)

    def _load_joblib(self, path: Path):
        return joblib.load(path) if path.exists() else None

    def _load_json(self, path: Path):
        if not path.exists():
            return None
        with open(path, "r") as f:
            return json.load(f)

    # =====================================================
    # MAIN LOAD
    # =====================================================

    def load_all(self) -> ModelArtifacts:

        artifacts = ModelArtifacts()

        # ---------------- MODELS ----------------
        artifacts.bias_model = self._load_torch_model(self.models_dir / "bias_model.pt")
        artifacts.ideology_model = self._load_torch_model(self.models_dir / "ideology_model.pt")
        artifacts.emotion_model = self._load_torch_model(self.models_dir / "emotion_model.pt")

        # ---------------- TOKENIZER ----------------
        tokenizer_path = self.models_dir / "tokenizer"
        if not tokenizer_path.exists():
            tokenizer_path = self.models_dir

        artifacts.tokenizer = self._load_tokenizer(tokenizer_path)

        # ---------------- FEATURES ----------------
        artifacts.feature_scaler = self._load_joblib(self.models_dir / "feature_scaler.pkl")
        artifacts.feature_selector = self._load_joblib(self.models_dir / "feature_selector.pkl")
        artifacts.feature_schema = self._load_json(self.models_dir / "feature_schema.json")

        # ---------------- METADATA ----------------
        artifacts.model_metadata = self.load_model_metadata()
        artifacts.model_config = self.load_model_config()

        # ---------------- PREDICTORS ----------------
        artifacts.bias_predictor = self._build_predictor(artifacts.bias_model)
        artifacts.ideology_predictor = self._build_predictor(artifacts.ideology_model)
        artifacts.emotion_predictor = self._build_predictor(artifacts.emotion_model)

        # ---------------- MULTITASK ----------------
        artifacts.multitask_model = self.load_multitask_model(artifacts.model_config)
        artifacts.multitask_predictor = self._build_predictor(artifacts.multitask_model)

        # ---------------- UNIFIED ----------------
        artifacts.unified_predictor = UnifiedPredictor(artifacts, self.device)

        return artifacts

    # =====================================================
    # BUILDERS
    # =====================================================

    def _build_predictor(self, model):

        if model is None:
            return None

        return Predictor(model=model, device=self.device)

    # =====================================================
    # MULTITASK
    # =====================================================

    def load_multitask_model(self, config):

        if config is None:
            return None

        model = ModelFactory.create_from_model_config(config)

        path = self.models_dir / "multitask_model.pt"

        if path.exists():
            state = torch.load(path, map_location="cpu")
            if isinstance(state, dict) and "state_dict" in state:
                state = state["state_dict"]
            model.load_state_dict(state)

        model.to(self.device)
        model.eval()

        return model

    # =====================================================
    # METADATA
    # =====================================================

    def load_model_metadata(self):

        path = self.models_dir / "metadata.json"

        if not path.exists():
            return None

        try:
            return ModelMetadata.load_json(path)
        except Exception:
            return None

    def load_model_config(self):

        for name in ["config.yaml", "model_config.yaml"]:
            path = self.models_dir / name
            if path.exists():
                try:
                    return ModelConfigLoader.load_multitask_config(path)
                except Exception:
                    return None

        return None

    # =====================================================
    # PUBLIC API
    # =====================================================

    def predict_for_evaluation(self, texts):

        artifacts = self.load_all()
        return artifacts.unified_predictor.predict_for_evaluation(texts)

    def get_model_versions(self):

        meta = self.load_model_metadata()

        if not meta:
            return {}

        return {
            "model_version": getattr(meta, "version", "unknown"),
            "trained_at": getattr(meta, "timestamp", None),
        }

    def validate_features(self, features: Dict[str, Any]):

        schema = self.load_all().feature_schema

        if not schema:
            return True

        missing = set(schema) - set(features)

        if missing:
            logger.warning(f"Missing features: {missing}")

        return True

    # =====================================================
    # ONNX EXPORT
    # =====================================================

    def export_onnx(self, model_name: str, output_path: str):

        model = self._load_torch_model(self.models_dir / f"{model_name}.pt")

        if model is None:
            raise ValueError(f"Model not found: {model_name}")

        dummy_ids = torch.ones(1, 16, dtype=torch.long).to(self.device)
        dummy_mask = torch.ones(1, 16, dtype=torch.long).to(self.device)

        class Wrapper(torch.nn.Module):
            def __init__(self, m):
                super().__init__()
                self.m = m

            def forward(self, input_ids, attention_mask):
                out = self.m(input_ids=input_ids, attention_mask=attention_mask)
                return out.logits if hasattr(out, "logits") else out

        torch.onnx.export(
            Wrapper(model),
            (dummy_ids, dummy_mask),
            output_path,
            input_names=["input_ids", "attention_mask"],
            output_names=["logits"],
            dynamic_axes={
                "input_ids": {0: "batch", 1: "seq"},
                "attention_mask": {0: "batch", 1: "seq"},
            },
            opset_version=17,
        )

        logger.info("ONNX exported to %s", output_path)