"""
File Name: predictor.py
Module: models.inference
Description:
    Provides a high-level prediction interface for TruthLens models.
    The Predictor class wraps model inference logic, handles device
    placement, batching, probability computation, and converts raw
    model outputs into standardized prediction dictionaries.

    The module is compatible with both single-task and multi-task
    models. For multi-task models, the predictor returns task-specific
    predictions and probabilities.

Dependencies:
    logging
    typing
    torch
    torch.nn
Inputs:
    Tokenized model inputs (input_ids, attention_mask)
Outputs:
    Structured prediction dictionaries
"""

from __future__ import annotations

import logging
from typing import Dict, Any, Optional

import numpy as np
import torch
import torch.nn as nn
from src.models.calibration import IsotonicCalibrator, TemperatureScaler

logger = logging.getLogger(__name__)


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

        self.device = torch.device(
            device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        )

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
    ) -> Dict[str, Any]:
        """
        Run prediction for a batch of inputs.
        """

        batch = self._move_to_device(batch)

        if self.ensemble_model is not None:
            outputs = self._run_ensemble(batch)
        else:
            outputs = self.model(**batch)

        return self._format_outputs(outputs)

    @torch.no_grad()
    def predict(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> Dict[str, Any]:
        """
        Run prediction for a single input example.
        """

        batch = {
            "input_ids": input_ids.unsqueeze(0),
            "attention_mask": attention_mask.unsqueeze(0),
        }

        results = self.predict_batch(batch)

        return self._squeeze_batch(results)

    def _format_outputs(
        self,
        outputs: Any,
    ) -> Dict[str, Any]:
        """
        Convert raw model outputs into prediction dictionary.
        """

        if isinstance(outputs, dict):

            formatted: Dict[str, Any] = {}

            for key, value in outputs.items():

                if isinstance(value, torch.Tensor):

                    if "logits" in key:
                        probs = torch.softmax(value, dim=-1)
                        calibrated_probs = self._calibrate_probabilities(
                            logits=value,
                            probabilities=probs,
                        )
                        preds = torch.argmax(calibrated_probs, dim=-1)

                        formatted[key.replace("logits", "predictions")] = preds
                        formatted[key.replace("logits", "probabilities")] = probs
                        formatted[
                            key.replace("logits", "calibrated_probabilities")
                        ] = calibrated_probs

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

        moved: Dict[str, torch.Tensor] = {}

        for k, v in batch.items():

            if isinstance(v, torch.Tensor):
                moved[k] = v.to(self.device)
            else:
                moved[k] = v

        return moved

    def _squeeze_batch(
        self,
        results: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Remove batch dimension for single predictions.
        """

        squeezed: Dict[str, Any] = {}

        for key, value in results.items():

            if isinstance(value, torch.Tensor) and value.size(0) == 1:
                squeezed[key] = value.squeeze(0)
            else:
                squeezed[key] = value

        return squeezed

    def _calibrate_probabilities(
        self,
        *,
        logits: torch.Tensor,
        probabilities: torch.Tensor,
    ) -> torch.Tensor:
        if self.temperature_scaler is not None:
            try:
                logits_device = logits.to(self.temperature_scaler.device)
                calibrated = self.temperature_scaler.predict_proba(logits_device)
                return calibrated.to(self.device)
            except Exception as exc:  # noqa: BLE001
                logger.warning("Temperature scaling skipped during prediction: %s", exc)

        if self.isotonic_calibrator is not None:
            try:
                probs_np = probabilities.detach().cpu().numpy().astype(np.float64)
                calibrated_np = self.isotonic_calibrator.predict_proba(probs_np)
                calibrated = torch.tensor(
                    calibrated_np,
                    dtype=probabilities.dtype,
                    device=probabilities.device,
                )
                return calibrated
            except Exception as exc:  # noqa: BLE001
                logger.warning("Isotonic calibration skipped during prediction: %s", exc)

        return probabilities

    def _run_ensemble(self, batch: Dict[str, torch.Tensor]) -> Dict[str, Any]:
        if self.ensemble_model is None:
            raise RuntimeError("Ensemble model is not configured.")

        logits: Optional[torch.Tensor] = None

        try:
            maybe_outputs = self.ensemble_model(**batch)
            if isinstance(maybe_outputs, torch.Tensor):
                logits = maybe_outputs
            elif isinstance(maybe_outputs, dict):
                for key, value in maybe_outputs.items():
                    if isinstance(value, torch.Tensor) and "logits" in key:
                        logits = value
                        break
        except Exception:  # noqa: BLE001
            logits = None

        if logits is None:
            if "ensemble_input" in batch and isinstance(batch["ensemble_input"], torch.Tensor):
                logits = self.ensemble_model(batch["ensemble_input"].to(self.device))
            elif "input_ids" in batch and isinstance(batch["input_ids"], torch.Tensor):
                logits = self.ensemble_model(batch["input_ids"].to(self.device))
            else:
                raise RuntimeError(
                    "Unable to run ensemble model: provide 'ensemble_input' "
                    "or compatible batch kwargs."
                )

        if not isinstance(logits, torch.Tensor):
            raise RuntimeError("Ensemble model must return logits tensor.")

        return {"ensemble_logits": logits}
