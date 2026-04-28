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

    # CRIT-1: previously lived in src.inference.inference_config and were
    # silently dropped at the engine boundary. Folded in here so the
    # inference loader and the engine speak the same dataclass.
    use_graph_analysis: bool = True
    cache_predictions: bool = False
    prediction_timeout: Optional[float] = None
    use_torch_compile: bool = False


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

        if self.temperature_scaler is None and self.isotonic_calibrator is None:
            # CRIT-7: previously the calibration code path swallowed every
            # error inside ``_apply_calibration`` and silently fell back to
            # raw softmax probabilities. Surface that fact at startup so
            # operators know calibrated probabilities are uncalibrated.
            logger.warning(
                "InferenceEngine: no calibrator attached — "
                "'calibrated_probabilities' will equal raw softmax probabilities."
            )

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
        """Single forward pass with a representative input.

        LAT-7: the previous warmup used a single token (``"warmup"``),
        which trained the cudnn autotuner on a one-token sequence and
        forced a re-tune on the first real request. We now use a longer,
        more representative string close to ``max_length`` so the first
        production request is not penalised.
        """
        try:
            target_tokens = max(64, min(self.config.max_length, 256))
            dummy_text = (" ".join(["warmup"] * target_tokens)).strip()
            self._forward([dummy_text])
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
        # CRIT-7: do not catch arbitrary exceptions silently. A broken
        # calibrator must surface as a real error during inference rather
        # than degrade to uncalibrated probabilities without any signal.
        if self.temperature_scaler:
            return self.temperature_scaler.predict_proba(logits)

        if self.isotonic_calibrator:
            return torch.tensor(
                self.isotonic_calibrator.predict_proba(
                    probs.detach().cpu().numpy()
                ),
                device=probs.device,
                dtype=probs.dtype,
            )

        return probs

    # =====================================================
    # 🔥 BASE INFERENCE
    # =====================================================
    #
    # CRIT-2: the previous return contract was a flat dict
    # ``{texts, predictions, probabilities, calibrated_probabilities, logits}``
    # while every downstream consumer (``run_inference``, predict_api's
    # ``predict_with_uncertainty``, ``prediction_service._compute_uncertainty``)
    # iterated over it as ``{task: {...}}``. We now return the nested
    # contract under the single task name ``"main"`` (the engine has one
    # classification head) and surface batch metadata under ``"_meta"``.

    DEFAULT_TASK_NAME = "main"

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

        task_output = {
            "predictions": preds,
            "probabilities": probs.numpy(),
            "calibrated_probabilities": cal.numpy(),
            "logits": logits.numpy(),
        }

        # CRIT-3/8: keep the engine's single-task output consistent with
        # the postprocessor's per-task contract. Calibrated probabilities
        # have already been computed above so we feed them in directly.
        try:
            postprocessed = self.postprocessor.process(
                {self.DEFAULT_TASK_NAME: {
                    "logits": task_output["logits"],
                    "probabilities": task_output["calibrated_probabilities"],
                }},
                task_types={self.DEFAULT_TASK_NAME: "multiclass"},
            )
            task_output.update({
                "labels": postprocessed[self.DEFAULT_TASK_NAME].get("labels"),
                "confidence": postprocessed[self.DEFAULT_TASK_NAME].get("confidence"),
            })
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("Postprocessor wiring skipped: %s", exc)

        return {
            self.DEFAULT_TASK_NAME: task_output,
            "_meta": {"texts": texts},
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

        task_out = outputs[self.DEFAULT_TASK_NAME]
        probs_arr = task_out["probabilities"]
        preds_arr = task_out["predictions"]

        # CRIT-4: ``fake_probability`` is only meaningful when the model's
        # label map is the legacy binary {0: real, 1: fake} contract. For
        # any other shape (>2 classes, missing label_map, or label names
        # that do not match the binary template) we emit ``None`` instead
        # of silently returning the prob of the second softmax slot.
        is_legacy_binary = self._is_legacy_binary_label_map(probs_arr.shape[-1])

        results = []
        for i, text in enumerate(texts):
            entry: Dict[str, Any] = {
                "text": text,
                "label": int(preds_arr[i]),
                "confidence": float(np.max(probs_arr[i])),
            }
            if is_legacy_binary:
                entry["fake_probability"] = float(probs_arr[i][1])
            else:
                entry["fake_probability"] = None
            results.append(entry)

        return results

    def _is_legacy_binary_label_map(self, num_classes: int) -> bool:
        if num_classes != 2:
            return False
        if not self.label_map:
            return False
        names = {str(v).lower() for v in self.label_map.values()}
        return {"real", "fake"}.issubset(names) or {"true", "fake"}.issubset(names)

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