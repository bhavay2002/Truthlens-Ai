from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Union, Any

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from src.models.calibration import IsotonicCalibrator, TemperatureScaler
from src.inference.postprocessing import Postprocessor


# DEV-2: AMP dtype must match training-time TRUTHLENS_AMP_DTYPE
# (default bf16). Hardcoding fp16 caused tail-class logit overflow
# for models trained with bf16. See inference_pipeline._resolve_amp_dtype
# for the canonical implementation; duplicated here to avoid an import
# cycle (pipeline imports from engine indirectly via prediction_service).
def _resolve_amp_dtype_engine() -> torch.dtype:
    requested = (os.environ.get("TRUTHLENS_AMP_DTYPE") or "bf16").lower()
    if requested in ("bf16", "bfloat16"):
        return torch.bfloat16
    if requested in ("fp16", "float16", "half"):
        return torch.float16
    return torch.float32

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

    # MEM-1: when True, accumulate per-batch logits/probs on the GPU and
    # transfer to CPU once at the end of ``predict_for_evaluation``. This
    # avoids per-batch host-buffer churn (the previous behaviour) on
    # short-to-medium evaluation runs. For very large evaluation passes,
    # leave at False to keep peak GPU memory bounded by one batch.
    keep_outputs_on_device: bool = False


# =========================================================
# ENGINE
# =========================================================

class InferenceEngine:

    def __init__(
        self,
        config: InferenceConfig,
        *,
        # MT-3: accept either a single calibrator (legacy single-task
        # contract) OR a per-task mapping ``{task: calibrator}``. A
        # multitask model needs one calibrator per head; the previous
        # single-object slot couldn't represent that.
        temperature_scaler: Optional[
            Union[TemperatureScaler, Mapping[str, TemperatureScaler]]
        ] = None,
        isotonic_calibrator: Optional[
            Union[IsotonicCalibrator, Mapping[str, IsotonicCalibrator]]
        ] = None,
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
        # DEV-2: dtype follows TRUTHLENS_AMP_DTYPE (default bf16) — the
        # same env the trainer reads. Hardcoding fp16 mismatched the
        # trainer's bf16 default and overflowed unbalanced-head logits.
        self.amp_dtype = (
            _resolve_amp_dtype_engine() if self.device.type == "cuda"
            else torch.float32
        )

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

        # DEV-1: load weights in fp32 even on CUDA. AMP autocast (used in
        # ``_forward``) gives the speed/memory win of fp16/bf16 compute
        # while keeping accumulators in fp32, which preserves the
        # calibration that was fit on fp32 logits during validation.
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path)

        self.model.to(self.device)
        self.model.eval()

        # DEV-3: freeze parameters once. ``.eval()`` only disables
        # Dropout / BatchNorm running stats; parameters still default to
        # ``requires_grad=True`` and pay autograd version-counter cost on
        # every forward pass (relevant when callers use ``no_grad``
        # rather than ``inference_mode``).
        for p in self.model.parameters():
            p.requires_grad_(False)

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

    def _resolve_calibrator(self, slot, task: str):
        """MT-3: pick the per-task calibrator from a Mapping, or return
        the single-object calibrator unchanged. Returns None when no
        calibrator is registered for the task.
        """
        if slot is None:
            return None
        # Treat any Mapping as per-task; sklearn calibrators don't
        # implement the Mapping protocol, so this is unambiguous.
        if isinstance(slot, Mapping):
            return slot.get(task)
        return slot

    def _apply_calibration(self, logits, probs, *, task: Optional[str] = None):
        # CRIT-7: do not catch arbitrary exceptions silently. A broken
        # calibrator must surface as a real error during inference rather
        # than degrade to uncalibrated probabilities without any signal.
        task_name = task or self.DEFAULT_TASK_NAME

        temp = self._resolve_calibrator(self.temperature_scaler, task_name)
        if temp is not None:
            return temp.predict_proba(logits)

        iso = self._resolve_calibrator(self.isotonic_calibrator, task_name)
        if iso is not None:
            # DEV-4: avoid the GPU→CPU→GPU→CPU ping-pong. The next
            # operation in the engine is ``cal.detach().cpu()`` so the
            # round-trip back to ``probs.device`` was pure waste.
            cal_np = iso.predict_proba(probs.detach().cpu().numpy())
            return torch.from_numpy(cal_np)

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

        # MEM-1: when ``keep_outputs_on_device`` is True, accumulate the
        # per-batch tensors on the GPU and transfer to CPU once at the
        # end. The previous code allocated a host buffer per batch via
        # ``.cpu()`` inside the loop, fragmenting host memory on long
        # eval runs. Keep the per-batch CPU path as the default so peak
        # GPU memory stays bounded by one batch.
        keep_on_device = bool(getattr(self.config, "keep_outputs_on_device", False))

        with torch.inference_mode():
            for batch in self._batchify(texts, self.config.batch_size):

                logits = self._forward(batch)
                probs = torch.softmax(logits, dim=-1)
                cal = self._apply_calibration(logits, probs)

                if keep_on_device:
                    all_logits.append(logits.detach())
                    all_probs.append(probs.detach())
                    # Calibration may already have moved cal off-device
                    # (DEV-4 isotonic path). Re-align to logits.device so
                    # the final ``cat`` has consistent placement.
                    cal_d = cal.detach()
                    if cal_d.device != logits.device:
                        cal_d = cal_d.to(logits.device)
                    all_cal.append(cal_d)
                else:
                    all_logits.append(logits.detach().cpu())
                    all_probs.append(probs.detach().cpu())
                    all_cal.append(cal.detach().cpu())

        logits = torch.cat(all_logits)
        probs = torch.cat(all_probs)
        cal = torch.cat(all_cal)

        if keep_on_device:
            logits = logits.cpu()
            probs = probs.cpu()
            cal = cal.cpu()

        preds = np.argmax(cal.numpy(), axis=1)

        task_output = {
            "predictions": preds,
            "probabilities": probs.numpy(),
            "calibrated_probabilities": cal.numpy(),
            "logits": logits.numpy(),
            # PP-4: surface task_type so PredictionService can pick the
            # correct entropy formula. The single-head HF engine is
            # multiclass by construction.
            "task_type": "multiclass",
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