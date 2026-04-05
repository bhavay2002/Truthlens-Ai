"""
File Name: model_wrapper.py
Module: models.inference
Description:
    Provides a standardized wrapper around TruthLens models to simplify
    inference, evaluation, and deployment. The wrapper abstracts device
    placement, model loading, checkpoint restoration, and unified forward
    execution.

    The wrapper is designed to work with both single-task models and the
    MultiTaskTruthLensModel. It exposes consistent methods for prediction,
    probability computation, and model checkpoint loading.

Dependencies:
    logging
    typing
    torch
    torch.nn
Inputs:
    Tokenized model inputs (input_ids, attention_mask)
Outputs:
    Model predictions and probabilities
"""

from __future__ import annotations

import logging
from typing import Dict, Optional, Any

import numpy as np
import torch
import torch.nn as nn
from src.models.calibration import IsotonicCalibrator, TemperatureScaler


logger = logging.getLogger(__name__)


class ModelWrapper:
    """
    Wrapper for managing model lifecycle and inference execution.
    """

    def __init__(
        self,
        model: nn.Module,
        device: Optional[str] = None,
        temperature_scaler: Optional[TemperatureScaler] = None,
        isotonic_calibrator: Optional[IsotonicCalibrator] = None,
        ensemble_model: Optional[nn.Module] = None,
    ) -> None:
        """
        Initialize model wrapper.

        Args:
            model:
                PyTorch model instance.
            device:
                Target device for inference ("cpu", "cuda", etc.)
        """

        if not isinstance(model, nn.Module):
            raise TypeError("model must be an instance of torch.nn.Module")

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

        logger.info("ModelWrapper initialized on device: %s", self.device)

    def set_temperature_scaler(self, scaler: TemperatureScaler) -> None:
        self.temperature_scaler = scaler

    def set_isotonic_calibrator(self, calibrator: IsotonicCalibrator) -> None:
        self.isotonic_calibrator = calibrator

    def set_ensemble_model(self, ensemble_model: nn.Module) -> None:
        self.ensemble_model = ensemble_model
        self.ensemble_model.to(self.device)
        self.ensemble_model.eval()

    def forward(
        self,
        batch: Dict[str, torch.Tensor],
    ) -> Dict[str, Any]:
        """
        Execute forward pass.

        Args:
            batch:
                Dictionary containing model inputs.

        Returns:
            Model outputs.
        """

        if not isinstance(batch, dict):
            raise TypeError("batch must be a dictionary")

        batch = self._move_to_device(batch)

        with torch.no_grad():
            if self.ensemble_model is not None:
                outputs = self._run_ensemble(batch)
            else:
                outputs = self.model(**batch)

        return outputs

    def predict(
        self,
        batch: Dict[str, torch.Tensor],
    ) -> Dict[str, Any]:
        """
        Run inference and return predictions.

        Args:
            batch:
                Input tensors.

        Returns:
            Prediction dictionary.
        """

        outputs = self.forward(batch)

        return self._extract_predictions(outputs)

    def load_checkpoint(
        self,
        checkpoint_path: str,
        strict: bool = True,
    ) -> None:
        """
        Load model weights from checkpoint.

        Args:
            checkpoint_path:
                Path to checkpoint file.
            strict:
                Whether to strictly enforce state dict keys.
        """

        try:
            checkpoint = torch.load(
                checkpoint_path,
                map_location=self.device,
                weights_only=False,
            )

            if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
                state_dict = checkpoint["model_state_dict"]
            else:
                state_dict = checkpoint

            self.model.load_state_dict(state_dict, strict=strict)

            logger.info("Checkpoint loaded successfully: %s", checkpoint_path)

        except Exception as exc:
            logger.exception("Failed to load checkpoint")
            raise RuntimeError("Checkpoint loading failed") from exc

    def _move_to_device(
        self,
        batch: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """
        Move batch tensors to device.
        """

        moved_batch: Dict[str, torch.Tensor] = {}

        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                moved_batch[key] = value.to(self.device)
            else:
                moved_batch[key] = value

        return moved_batch

    def _extract_predictions(
        self,
        outputs: Any,
    ) -> Dict[str, Any]:
        """
        Convert model outputs to prediction format.
        """

        if isinstance(outputs, dict):

            results: Dict[str, Any] = {}

            for key, value in outputs.items():

                if isinstance(value, torch.Tensor):

                    if "logits" in key:

                        probs = torch.softmax(value, dim=-1)
                        calibrated_probs = self._calibrate_probabilities(
                            logits=value,
                            probabilities=probs,
                        )
                        preds = torch.argmax(calibrated_probs, dim=-1)

                        results[key.replace("logits", "probabilities")] = probs
                        results[
                            key.replace("logits", "calibrated_probabilities")
                        ] = calibrated_probs
                        results[key.replace("logits", "predictions")] = preds

                    else:
                        results[key] = value

                else:
                    results[key] = value

            return results

        raise RuntimeError("Unsupported model output format")

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
                return calibrated.to(probabilities.device)
            except Exception as exc:  # noqa: BLE001
                logger.warning("Temperature scaling skipped during wrapper prediction: %s", exc)

        if self.isotonic_calibrator is not None:
            try:
                probs_np = probabilities.detach().cpu().numpy().astype(np.float64)
                calibrated_np = self.isotonic_calibrator.predict_proba(probs_np)
                return torch.tensor(
                    calibrated_np,
                    dtype=probabilities.dtype,
                    device=probabilities.device,
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning("Isotonic calibration skipped during wrapper prediction: %s", exc)

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
