from __future__ import annotations

import logging
from typing import Dict, Any, Optional

import numpy as np
import torch
import torch.nn as nn

from src.models.calibration import IsotonicCalibrator, TemperatureScaler
from src.models.inference.prediction_output import PredictionOutput
from src.models.multitask.multitask_output import MultiTaskOutput
from src.utils import get_device, move_to_device

logger = logging.getLogger(__name__)

DEFAULT_FAKE_INDEX = 1
FAKE_LABEL_CANDIDATES = {"fake", "false", "misleading"}

_FAKE_HEAD_KEYS = (
    "fake_logits",
    "fakenews_logits",
    "misinformation_logits",
)


def _find_tensor_by_keys(
    data: Dict[str, Any],
    keys: tuple[str, ...],
) -> Optional[torch.Tensor]:
    for key in keys:
        value = data.get(key)
        if isinstance(value, torch.Tensor):
            return value
    return None


class Predictor:
    """
    Production-grade predictor wrapper for TruthLens models.
    """

    def __init__(
        self,
        model: nn.Module,
        device: str | None = None,
        temperature_scaler: Optional[TemperatureScaler] = None,
        isotonic_calibrator: Optional[IsotonicCalibrator] = None,
        ensemble_model: Optional[nn.Module] = None,
    ) -> None:

        if not isinstance(model, nn.Module):
            raise TypeError("model must be torch.nn.Module")

        self.model = model
        self.device = torch.device(device) if device else get_device(prefer_gpu=True)

        self.model.to(self.device)
        self.model.eval()

        self.ensemble_model = ensemble_model
        if self.ensemble_model is not None:
            self.ensemble_model.to(self.device)
            self.ensemble_model.eval()

        self.temperature_scaler = temperature_scaler
        self.isotonic_calibrator = isotonic_calibrator

        logger.info("Predictor initialized on device %s", self.device)

    # =========================================================
    #  PUBLIC API
    # =========================================================

    def set_temperature_scaler(self, scaler: TemperatureScaler) -> None:
        self.temperature_scaler = scaler

    def set_isotonic_calibrator(self, calibrator: IsotonicCalibrator) -> None:
        self.isotonic_calibrator = calibrator

    def set_ensemble_model(self, ensemble_model: nn.Module) -> None:
        self.ensemble_model = ensemble_model
        self.ensemble_model.to(self.device)
        self.ensemble_model.eval()

    # =========================================================
    #  INFERENCE
    # =========================================================

    @torch.inference_mode()
    def predict_batch(
        self,
        batch: Dict[str, torch.Tensor],
        *,
        return_structured: bool = False,
    ) -> Dict[str, Any]:

        batch = move_to_device(batch, self.device)

        outputs = self._forward(batch)
        formatted = self._format_outputs(outputs)

        if return_structured:
            return PredictionOutput.from_raw_outputs(formatted).to_dict()

        return formatted

    @torch.inference_mode()
    def predict(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        *,
        return_structured: bool = False,
    ) -> Dict[str, Any]:

        batch = {
            "input_ids": input_ids.unsqueeze(0),
            "attention_mask": attention_mask.unsqueeze(0),
        }

        results = self.predict_batch(batch, return_structured=return_structured)
        return self._squeeze_batch(results)

    @torch.inference_mode()
    def predict_batch_structured(
        self,
        batch: Dict[str, torch.Tensor],
    ) -> PredictionOutput:
        outputs = self.predict_batch(batch)
        return PredictionOutput.from_raw_outputs(outputs)

    @torch.inference_mode()
    def predict_batch_pairs(
        self,
        batch: Dict[str, torch.Tensor],
    ) -> list[list[float]]:

        batch = move_to_device(batch, self.device)
        outputs = self._forward(batch)
        formatted = self._format_outputs(outputs)

        n = self._infer_batch_size(formatted)

        results: list[list[float]] = []

        for i in range(n):
            sample = {
                k: (
                    v[i].unsqueeze(0)
                    if isinstance(v, torch.Tensor) and v.dim() > 0 and v.size(0) == n
                    else v
                )
                for k, v in formatted.items()
            }

            output = self.build_fake_real_output(sample)

            fake_prob = output["fake_probability"]
            real_prob = 1.0 - fake_prob

            results.append([round(real_prob, 6), round(fake_prob, 6)])

        return results

    # =========================================================
    #  CORE FORWARD
    # =========================================================

    def _forward(self, batch: Dict[str, torch.Tensor]) -> Any:

        use_amp = self.device.type == "cuda"

        if self.ensemble_model is not None:
            return self._run_ensemble(batch)

        if use_amp:
            amp_dtype = (
                torch.bfloat16
                if self.device.type == "cuda" and torch.cuda.is_bf16_supported()
                else torch.float16
            )
            with torch.autocast(device_type=self.device.type, dtype=amp_dtype):
                return self.model(**batch)
        else:
            return self.model(**batch)

    # =========================================================
    #  OUTPUT FORMATTING
    # =========================================================

    def _format_outputs(self, outputs: Any) -> Dict[str, Any]:

        if isinstance(outputs, MultiTaskOutput):
            return outputs.to_flat_prediction_dict()

        if isinstance(outputs, dict):

            multitask_output = outputs.get("multitask_output")
            if isinstance(multitask_output, MultiTaskOutput):
                return multitask_output.to_flat_prediction_dict()

            formatted: Dict[str, Any] = {}

            for key, value in outputs.items():

                if (
                    isinstance(value, torch.Tensor)
                    and key.endswith("_logits")
                    and value.dim() >= 2
                    and value.size(-1) >= 2
                ):
                    logits = torch.nan_to_num(value, nan=0.0, posinf=1e6, neginf=-1e6)

                    # torch.softmax is already numerically stable (internally
                    # subtracts the per-row max), so an explicit max-subtract
                    # here is redundant work on every batch.
                    probs = torch.softmax(logits, dim=-1)

                    calibrated = self._calibrate_probabilities(
                        logits=logits,
                        probabilities=probs,
                    )

                    preds = torch.argmax(probs, dim=-1)

                    # Precise suffix strip: since key endswith("_logits"),
                    # removesuffix is safe and cannot mangle other substrings.
                    base = key[: -len("_logits")]
                    formatted[key] = logits
                    formatted[f"{base}_probabilities"] = calibrated
                    formatted[f"{base}_predictions"] = preds

                else:
                    formatted[key] = value

            return formatted

        raise RuntimeError("Unsupported model output format")

    # =========================================================
    #  CALIBRATION
    # =========================================================

    def _calibrate_probabilities(
        self,
        *,
        logits: torch.Tensor,
        probabilities: torch.Tensor,
    ) -> torch.Tensor:

        if self.temperature_scaler is not None:
            try:
                scaler_device = getattr(self.temperature_scaler, "device", self.device)
                logits_device = logits.to(scaler_device)
                calibrated = self.temperature_scaler.predict_proba(logits_device)
                return calibrated.to(self.device, dtype=probabilities.dtype)
            except Exception as exc:
                logger.warning("Temperature scaling skipped: %s", exc)

        if self.isotonic_calibrator is not None:
            try:
                probs_cpu = probabilities.detach().to("cpu", non_blocking=True)
                probs_np = probs_cpu.numpy().astype(np.float64)

                calibrated_np = self.isotonic_calibrator.predict_proba(probs_np)

                return torch.tensor(
                    calibrated_np,
                    dtype=probabilities.dtype,
                    device=self.device,
                )
            except Exception as exc:
                logger.warning("Isotonic calibration skipped: %s", exc)

        return probabilities

    # =========================================================
    # 🔧 FAKE/REAL LOGIC (STRICT)
    # =========================================================

    def build_fake_real_output(self, formatted: Dict[str, Any]) -> Dict[str, Any]:

        probs = self._extract_fake_probs(formatted)

        if probs is None:
            raise RuntimeError(
                "Model returned no fake/misinformation head; refusing to fabricate.\n"
                f"Available outputs: {list(formatted.keys())}"
            )

        fake_index = self._resolve_fake_index()

        if probs.dim() > 1:
            mean_probs = probs.mean(dim=0)
        else:
            mean_probs = probs

        fake_prob = float(mean_probs[fake_index].item())
        confidence = float(mean_probs.max().item())

        num_classes = mean_probs.size(-1)

        if num_classes == 2:
            pred_index = fake_index if fake_prob >= 0.5 else (1 - fake_index)
        else:
            pred_index = int(mean_probs.argmax().item())

        label = "Fake" if pred_index == fake_index else "Real"

        return {
            "label": label,
            "fake_probability": float(min(max(fake_prob, 0.0), 1.0)),
            "confidence": float(min(max(confidence, 0.0), 1.0)),
        }

    def _extract_fake_probs(self, formatted: Dict[str, Any]) -> Optional[torch.Tensor]:

        # 1) Named multi-task heads (fake_logits / fakenews_logits / ...)
        for key in _FAKE_HEAD_KEYS:
            probs_key = key[: -len("_logits")] + "_probabilities"
            probs = formatted.get(probs_key)
            if isinstance(probs, torch.Tensor):
                if probs.numel() == 0:
                    raise RuntimeError("Empty probability tensor from fake head")
                return probs

        logits = _find_tensor_by_keys(formatted, _FAKE_HEAD_KEYS)

        if isinstance(logits, torch.Tensor):
            logits = torch.nan_to_num(logits, nan=0.0, posinf=1e6, neginf=-1e6)
            if logits.numel() == 0:
                raise RuntimeError("Empty probability tensor from fake head")
            # torch.softmax is numerically stable; redundant max-subtract removed.
            return torch.softmax(logits, dim=-1)

        # 2) Plain HuggingFace-style single-head classifier: the model emits
        #    a single "logits" tensor and its config.id2label declares a
        #    Fake/Real (or equivalent) mapping. Use it directly instead of
        #    refusing — this honors the actual model config, not a fabricated
        #    head. Guarded by config inspection so we do NOT coerce unrelated
        #    classifiers into fake/real semantics.
        plain_logits = formatted.get("logits")
        if isinstance(plain_logits, torch.Tensor) and plain_logits.dim() >= 2 and plain_logits.size(-1) >= 2:
            config = getattr(self.model, "config", None)
            id2label = getattr(config, "id2label", None) if config is not None else None
            if isinstance(id2label, dict):
                labels_lower = {str(v).lower() for v in id2label.values()}
                if labels_lower & FAKE_LABEL_CANDIDATES:
                    logits = torch.nan_to_num(
                        plain_logits, nan=0.0, posinf=1e6, neginf=-1e6
                    )
                    return torch.softmax(logits, dim=-1)

        return None

    def _resolve_fake_index(self) -> int:

        config = getattr(self.model, "config", None)

        if config is not None:

            id2label = getattr(config, "id2label", None)
            if isinstance(id2label, dict):
                for idx, label in id2label.items():
                    if isinstance(label, str) and label.lower() in FAKE_LABEL_CANDIDATES:
                        return int(idx)

            label2id = getattr(config, "label2id", None)
            if isinstance(label2id, dict):
                for label, idx in label2id.items():
                    if isinstance(label, str) and label.lower() in FAKE_LABEL_CANDIDATES:
                        return int(idx)

        logger.warning("Using default fake index: %d", DEFAULT_FAKE_INDEX)
        if DEFAULT_FAKE_INDEX < 0:
            raise RuntimeError("Invalid fake index")
        return DEFAULT_FAKE_INDEX

    # =========================================================
    #  UTIL
    # =========================================================

    def _infer_batch_size(self, formatted: Dict[str, Any]) -> int:
        for value in formatted.values():
            if isinstance(value, torch.Tensor) and value.dim() > 1:
                return value.size(0)
        raise RuntimeError("Cannot infer batch size")

    def _squeeze_batch(self, results: Dict[str, Any]) -> Dict[str, Any]:

        return {
            k: v.squeeze(0) if isinstance(v, torch.Tensor) and v.dim() > 0 and v.size(0) == 1 else v
            for k, v in results.items()
        }

    def _run_ensemble(self, batch: Dict[str, torch.Tensor]) -> Dict[str, Any]:

        outputs = self.ensemble_model(**batch)

        if isinstance(outputs, torch.Tensor):
            logits = outputs

        elif isinstance(outputs, dict):

            if "logits" in outputs:
                logits = outputs["logits"]

            elif "fake_logits" in outputs:
                logits = outputs["fake_logits"]

            else:
                raise RuntimeError(
                    "Ensemble must return 'logits' or 'fake_logits'. "
                    f"Available keys: {list(outputs.keys())}"
                )

        else:
            raise RuntimeError("Invalid ensemble output format")

        return {"ensemble_logits": logits}