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

_PROPAGANDA_KEYS = (
    "propaganda_logits",
    "propaganda_predictions",
    "propaganda_probabilities",
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
    Generic predictor wrapper for TruthLens models.
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

    def set_temperature_scaler(self, scaler: TemperatureScaler) -> None:
        self.temperature_scaler = scaler

    def set_isotonic_calibrator(self, calibrator: IsotonicCalibrator) -> None:
        self.isotonic_calibrator = calibrator

    def set_ensemble_model(self, ensemble_model: nn.Module) -> None:
        self.ensemble_model = ensemble_model
        self.ensemble_model.to(self.device)
        self.ensemble_model.eval()

    @torch.no_grad()
    def predict_batch(
        self,
        batch: Dict[str, torch.Tensor],
        *,
        return_structured: bool = False,
    ) -> Dict[str, Any]:
        """
        Run prediction for a batch of inputs.
        """

        batch = self._move_to_device(batch)

        if self.ensemble_model is not None:
            outputs = self._run_ensemble(batch)
        else:
            outputs = self.model(**batch)

        formatted = self._format_outputs(outputs)
        if return_structured:
            return PredictionOutput.from_raw_outputs(formatted).to_dict()
        return formatted

    @torch.no_grad()
    def predict(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        *,
        return_structured: bool = False,
    ) -> Dict[str, Any]:
        """
        Run prediction for a single input example.
        """

        batch = {
            "input_ids": input_ids.unsqueeze(0),
            "attention_mask": attention_mask.unsqueeze(0),
        }

        results = self.predict_batch(batch, return_structured=return_structured)

        return self._squeeze_batch(results)

    @torch.no_grad()
    def predict_batch_structured(
        self,
        batch: Dict[str, torch.Tensor],
    ) -> PredictionOutput:
        outputs = self.predict_batch(batch)
        return PredictionOutput.from_raw_outputs(outputs)

    def _format_outputs(
        self,
        outputs: Any,
    ) -> Dict[str, Any]:
        """
        Convert raw model outputs into prediction dictionary.
        """

        if isinstance(outputs, MultiTaskOutput):
            return outputs.to_flat_prediction_dict()

        if isinstance(outputs, dict):

            multitask_output = outputs.get("multitask_output")
            if isinstance(multitask_output, MultiTaskOutput):
                return multitask_output.to_flat_prediction_dict()

            has_nested_task_dicts = any(
                isinstance(value, dict) and isinstance(value.get("logits"), torch.Tensor)
                for value in outputs.values()
            )
            if has_nested_task_dicts:
                return MultiTaskOutput.from_model_outputs(outputs).to_flat_prediction_dict()

            formatted: Dict[str, Any] = {}

            for key, value in outputs.items():

                if isinstance(value, torch.Tensor):

                    if "logits" in key:
                        logits = value
                        logits = torch.nan_to_num(
                            logits,
                            nan=0.0,
                            posinf=1e6,
                            neginf=-1e6,
                        )
                        probs = torch.softmax(logits, dim=-1)
                        calibrated_probs = self._calibrate_probabilities(
                            logits=logits,
                            probabilities=probs,
                        )
                        preds = torch.argmax(probs, dim=-1)

                        formatted[key] = logits
                        formatted[key.replace("logits", "predictions")] = preds
                        formatted[key.replace("logits", "probabilities")] = calibrated_probs

                    else:
                        formatted[key] = value

                else:
                    formatted[key] = value

            return formatted

        raise RuntimeError("Unsupported model output format")

    def _move_to_device(
        self,
        batch: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:

        return move_to_device(batch, self.device)

    def _squeeze_batch(
        self,
        results: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Remove batch dimension for single predictions.
        """

        squeezed: Dict[str, Any] = {}

        for key, value in results.items():

            if isinstance(value, torch.Tensor):
                if value.dim() > 0 and value.size(0) == 1:
                    squeezed[key] = value.squeeze(0)
                else:
                    squeezed[key] = value
            else:
                squeezed[key] = value

        return squeezed

    def _calibrate_probabilities(
        self,
        *,
        logits: torch.Tensor,
        probabilities: torch.Tensor,
    ) -> torch.Tensor:
        logits = logits.to(self.device)
        probabilities = probabilities.to(self.device)

        if self.temperature_scaler is not None:
            try:
                logits_device = logits.to(self.temperature_scaler.device)
                calibrated = self.temperature_scaler.predict_proba(logits_device)
                return calibrated.to(self.device, dtype=probabilities.dtype)
            except Exception as exc:  # noqa: BLE001
                logger.warning("Temperature scaling skipped: %s", exc)

        if self.isotonic_calibrator is not None:
            try:
                probs_np = probabilities.detach().cpu().numpy().astype(np.float64)
                calibrated_np = self.isotonic_calibrator.predict_proba(probs_np)
                calibrated = torch.tensor(
                    calibrated_np,
                    dtype=probabilities.dtype,
                    device=self.device,
                )
                return calibrated
            except Exception as exc:  # noqa: BLE001
                logger.warning("Isotonic calibration skipped: %s", exc)

        return probabilities

    def _resolve_fake_index(self) -> int:
        """
        Resolve the index corresponding to the 'FAKE' class.

        Falls back to DEFAULT_FAKE_INDEX if not found, with warning.
        """

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

        logger.warning(
            "Could not resolve 'FAKE' index from model config; "
            "defaulting to index %d. Verify label ordering.",
            DEFAULT_FAKE_INDEX,
        )

        return DEFAULT_FAKE_INDEX

    def _extract_fake_probs(
        self,
        formatted: Dict[str, Any],
    ) -> Optional[torch.Tensor]:
        """
        Return Fake/Real probabilities if a dedicated head exists.
        Never falls back to propaganda.
        """

        for key in _FAKE_HEAD_KEYS:
            probs = formatted.get(key.replace("logits", "probabilities"))
            if isinstance(probs, torch.Tensor):
                return probs

        logits = _find_tensor_by_keys(formatted, _FAKE_HEAD_KEYS)

        if isinstance(logits, torch.Tensor):
            logits = torch.nan_to_num(
                logits,
                nan=0.0,
                posinf=1e6,
                neginf=-1e6,
            )
            return torch.softmax(logits, dim=-1)

        return None

    def _compose_fake_probability(self, formatted: Dict[str, Any]) -> float:
        """
        Compose Fake probability from multiple non-propaganda signals.
        """

        signals: list[float] = []

        def _safe_positive_class(tensor: torch.Tensor) -> float:
            if tensor.dim() == 0:
                return float(tensor.item())
            return float(tensor[..., -1].mean().item())

        bias = formatted.get("bias_probabilities")
        if isinstance(bias, torch.Tensor):
            signals.append(_safe_positive_class(bias))

        emotion = formatted.get("emotion_probabilities")
        if isinstance(emotion, torch.Tensor):
            signals.append(_safe_positive_class(emotion) * 0.5)

        ideology = formatted.get("ideology_probabilities")
        if isinstance(ideology, torch.Tensor):
            signals.append(_safe_positive_class(ideology) * 0.5)

        if not signals:
            return 0.5

        score = sum(signals) / len(signals)
        return float(min(max(score, 0.0), 1.0))

    def build_fake_real_output(self, formatted: Dict[str, Any]) -> Dict[str, Any]:
        """
        Build Fake/Real outputs using a dedicated head or composed fallback.
        """

        probs = self._extract_fake_probs(formatted)
        fake_index = self._resolve_fake_index()

        if probs is not None:
            if probs.dim() > 1:
                if probs.size(0) > 1:
                    logger.warning(
                        "build_fake_real_output received batch size > 1; "
                        "aggregating via mean (consider per-sample handling upstream)"
                    )
                mean_probs = probs.mean(dim=0)
            else:
                mean_probs = probs
            fake_prob = float(mean_probs[fake_index].item())
            confidence = float(mean_probs.max().item())
            pred_index = int(mean_probs.argmax().item())
        else:
            fake_prob = self._compose_fake_probability(formatted)
            confidence = float(fake_prob)
            pred_index = int(fake_prob >= 0.5)

        label = "Fake" if pred_index == fake_index else "Real"

        return {
            "label": label,
            "fake_probability": float(min(max(fake_prob, 0.0), 1.0)),
            "confidence": float(min(max(confidence, 0.0), 1.0)),
        }

    @torch.no_grad()
    def predict_batch_pairs(
        self,
        batch: Dict[str, torch.Tensor],
    ) -> list[list[float]]:
        """
        Return per-sample [real_prob, fake_prob] pairs for a batch.

        - Runs one forward pass
        - Splits outputs per sample safely
        - Uses build_fake_real_output per sample (no cross-sample averaging)
        """

        batch = self._move_to_device(batch)

        if self.ensemble_model is not None:
            outputs = self._run_ensemble(batch)
        else:
            outputs = self.model(**batch)

        formatted = self._format_outputs(outputs)

        n = None
        for value in formatted.values():
            if isinstance(value, torch.Tensor) and value.dim() > 0:
                n = value.size(0)
                break
        if n is None:
            raise RuntimeError("Cannot infer batch size from outputs")

        results: list[list[float]] = []
        for i in range(n):
            sample = {
                key: (
                    value[i : i + 1]
                    if isinstance(value, torch.Tensor)
                    and value.dim() > 0
                    and value.size(0) == n
                    else value
                )
                for key, value in formatted.items()
            }

            output = self.build_fake_real_output(sample)
            fake_prob = float(output["fake_probability"])
            real_prob = float(1.0 - fake_prob)

            results.append([round(real_prob, 6), round(fake_prob, 6)])

        return results

    def _validate_binary_logits(self, logits: torch.Tensor) -> None:
        if logits.size(-1) != 2:
            logger.warning(
                "Expected binary classification logits (size=2), got size=%d",
                logits.size(-1),
            )

    def _run_ensemble(self, batch: Dict[str, torch.Tensor]) -> Dict[str, Any]:
        if self.ensemble_model is None:
            raise RuntimeError("Ensemble model is not configured.")

        try:
            outputs = self.ensemble_model(**batch)
        except Exception as exc:  # noqa: BLE001
            logger.error("Ensemble model failed: %s", exc)
            raise RuntimeError("Ensemble model execution failed") from exc

        if isinstance(outputs, torch.Tensor):
            logits = outputs
        elif isinstance(outputs, dict):
            logits = next(
                (
                    v 
                    for k, v in outputs.items()
                    if isinstance(v, torch.Tensor) and "logits" in k
                ),
                None,
            )
            if logits is None:
                logits = next(
                    (v for v in outputs.values() if isinstance(v, torch.Tensor)),
                    None,
                )
        else:
            raise RuntimeError("Invalid ensemble output format")

        if not isinstance(logits, torch.Tensor):
            raise RuntimeError("Ensemble model must return logits tensor")

        return {"ensemble_logits": logits}
