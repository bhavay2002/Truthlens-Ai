from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Union, Any

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from src.models.calibration import IsotonicCalibrator, TemperatureScaler
from src.inference.postprocessing import Postprocessor

# 🔥 NEW IMPORTS
# NOTE: PredictionService is imported lazily inside ``InferenceEngine.__init__``
# to avoid the circular import (prediction_service ← inference_engine).
from src.inference.schema import PredictionOutput

from src.utils import (
    ensure_file_exists,
    ensure_non_empty_text_list,
    get_device,
    load_json,
)

logger = logging.getLogger(__name__)


# =========================================================
# CONFIG
# =========================================================

@dataclass
class InferenceConfig:
    model_path: str
    tokenizer_path: Optional[str] = None
    device: str = "auto"
    max_length: int = 512
    batch_size: int = 16
    return_probabilities: bool = True
    return_logits: bool = True
    use_amp: bool = True

    # 🔥 NEW
    enable_full_pipeline: bool = True


# =========================================================
# ENGINE
# =========================================================

class InferenceEngine:

    def __init__(
        self,
        config: InferenceConfig,
        *,
        temperature_scaler: Optional[TemperatureScaler] = None,
        isotonic_calibrator: Optional[IsotonicCalibrator] = None,
        postprocessor: Optional[Postprocessor] = None,
    ):
        self.config = config
        self.device = self._resolve_device(config.device)

        self.temperature_scaler = temperature_scaler
        self.isotonic_calibrator = isotonic_calibrator

        self.model = None
        self.tokenizer = None
        self.label_map: Optional[Dict[int, str]] = None

        self.postprocessor = postprocessor or Postprocessor()

        self.use_amp = self.device.type == "cuda" and config.use_amp
        self.amp_dtype = torch.float16 if self.device.type == "cuda" else torch.float32

        self._load_model()

        # 🔥 NEW: Prediction Service (FULL SYSTEM)
        self.prediction_service = None

        if config.enable_full_pipeline:
            # Lazy import to break circular dependency.
            from src.inference.prediction_service import PredictionService

            self.prediction_service = PredictionService(
                engine=self,
            )

    # =====================================================
    # DEVICE
    # =====================================================

    def _resolve_device(self, device: str) -> torch.device:
        if device == "auto":
            return get_device(prefer_gpu=True)
        return torch.device(device)

    # =====================================================
    # MODEL LOAD
    # =====================================================

    def _load_model(self):

        model_path = Path(self.config.model_path)
        ensure_file_exists(model_path)

        tokenizer_path = self.config.tokenizer_path or self.config.model_path

        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, use_fast=True)

        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_path,
            torch_dtype=self.amp_dtype if self.device.type == "cuda" else None,
        )

        self.model.to(self.device)
        self.model.eval()

        self._load_label_map(model_path)
        self._warmup()

    def _load_label_map(self, path: Path):
        file = path / "label_map.json"
        if file.exists():
            raw = load_json(file)
            self.label_map = {int(k): v for k, v in raw.items()}

    def _warmup(self):
        """Single forward pass with a dummy input to trigger JIT/cudnn autotuning."""
        try:
            dummy = ["warmup"]
            self._forward(dummy)
            logger.info("InferenceEngine warmup complete (device=%s)", self.device)
        except Exception as exc:
            logger.debug("InferenceEngine warmup skipped: %s", exc)

    # =====================================================
    # HELPERS
    # =====================================================

    def _validate_input(self, texts):
        if isinstance(texts, str):
            texts = [texts]
        return ensure_non_empty_text_list(texts, "texts")

    def _batchify(self, items, size):
        return [items[i:i + size] for i in range(0, len(items), size)]

    # =====================================================
    # CORE FORWARD
    # =====================================================

    def _forward(self, batch):

        encoded = self.tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=self.config.max_length,
            return_tensors="pt",
        )
        encoded = {k: v.to(self.device, non_blocking=True) for k, v in encoded.items()}

        if self.use_amp:
            with torch.autocast(device_type=self.device.type, dtype=self.amp_dtype):
                logits = self.model(**encoded).logits
        else:
            logits = self.model(**encoded).logits

        return logits

    # =====================================================
    # CALIBRATION
    # =====================================================

    def _apply_calibration(self, logits, probs):

        if self.temperature_scaler:
            try:
                return self.temperature_scaler.predict_proba(logits)
            except Exception:
                pass

        if self.isotonic_calibrator:
            try:
                return torch.tensor(
                    self.isotonic_calibrator.predict_proba(
                        probs.detach().cpu().numpy()
                    ),
                    device=probs.device,
                    dtype=probs.dtype,
                )
            except Exception:
                pass

        return probs

    # =====================================================
    # 🔥 BASE INFERENCE (UNCHANGED)
    # =====================================================

    def predict_for_evaluation(
        self,
        texts: Union[str, List[str]],
    ) -> Dict[str, Any]:

        texts = self._validate_input(texts)

        all_logits = []
        all_probs = []
        all_cal = []

        with torch.inference_mode():
            for batch in self._batchify(texts, self.config.batch_size):

                logits = self._forward(batch)
                probs = torch.softmax(logits, dim=-1)
                cal = self._apply_calibration(logits, probs)

                all_logits.append(logits.detach().cpu())
                all_probs.append(probs.detach().cpu())
                all_cal.append(cal.detach().cpu())

        logits = torch.cat(all_logits)
        probs = torch.cat(all_probs)
        cal = torch.cat(all_cal)

        preds = np.argmax(cal.numpy(), axis=1)

        return {
            "texts": texts,
            "predictions": preds,
            "probabilities": probs.numpy(),
            "calibrated_probabilities": cal.numpy(),
            "logits": logits.numpy(),
        }

    # =====================================================
    # 🔥 NEW FULL PIPELINE
    # =====================================================

    def predict_full(self, text: str) -> Dict[str, Any]:
        """
        🔥 FULL SYSTEM:
        model + graph + explainability + aggregation +
        postprocessing + drift + monitoring + schema
        """

        if not self.prediction_service:
            raise RuntimeError("Full pipeline not enabled")

        result = self.prediction_service.predict(text)

        # optional schema validation
        try:
            result = PredictionOutput(**result).model_dump()
        except Exception as e:
            logger.warning("Schema validation failed: %s", e)

        return result

    # =====================================================
    # 🔥 LEGACY USER API
    # =====================================================

    def predict(self, texts):
        texts = self._validate_input(texts)
        outputs = self.predict_for_evaluation(texts)

        results = []
        probs_arr = outputs["probabilities"]
        preds_arr = outputs["predictions"]

        for i, text in enumerate(texts):
            results.append({
                "text": text,
                "label": int(preds_arr[i]),
                "confidence": float(np.max(probs_arr[i])),
                "fake_probability": float(probs_arr[i][1]) if probs_arr.shape[1] > 1 else float(probs_arr[i][0]),
            })

        return results

    def predict_single(self, text: str):
        return self.predict([text])[0]

    # =====================================================
    # INFO
    # =====================================================

    def get_model_info(self):
        return {
            "device": str(self.device),
            "params": sum(p.numel() for p in self.model.parameters()),
            "full_pipeline_enabled": self.prediction_service is not None,
        }