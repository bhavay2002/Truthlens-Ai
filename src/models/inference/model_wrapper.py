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
from src.models.inference.prediction_output import PredictionOutput


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
        *,
        use_half_precision: bool = False,
        compile_mode: Optional[str] = None,
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
        self.compute_probabilities = False
        self.return_logits_only = False

        if use_half_precision and self.device.type == "cuda":
            self.model.half()
            if self.ensemble_model is not None:
                self.ensemble_model.half()

        if compile_mode and hasattr(torch, "compile"):
            try:
                self.model = torch.compile(self.model, mode=compile_mode)
            except Exception as exc:  # noqa: BLE001
                logger.warning("Model compilation skipped: %s", exc)

            if self.ensemble_model is not None:
                try:
                    self.ensemble_model = torch.compile(
                        self.ensemble_model,
                        mode=compile_mode,
                    )
                except Exception as exc:  # noqa: BLE001
                    logger.warning("Ensemble model compilation skipped: %s", exc)

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

        if self.device.type == "cuda":
            autocast_dtype = (
                torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
            )
        else:
            autocast_dtype = torch.float32

        with torch.inference_mode():
            with torch.autocast(
                device_type=self.device.type,
                dtype=autocast_dtype,
                enabled=self.device.type == "cuda",
            ):
                if self.ensemble_model is not None:
                    outputs = self._run_ensemble(batch)
                else:
                    outputs = self.model(**batch)

        return outputs

    def predict(
        self,
        batch: Dict[str, torch.Tensor],
        *,
        return_structured: bool = False,
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
        extracted = self._extract_predictions(outputs)

        if return_structured:
            return PredictionOutput.from_raw_outputs(extracted).to_dict()

        return extracted

    def predict_structured(
        self,
        batch: Dict[str, torch.Tensor],
    ) -> PredictionOutput:
        outputs = self.forward(batch)
        extracted = self._extract_predictions(outputs)
        return PredictionOutput.from_raw_outputs(extracted)

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

        except Exception as exc:
            raise RuntimeError("Checkpoint loading failed") from exc

    def _move_to_device(
        self,
        batch: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """
        Move batch tensors to device.

        Pinned memory + non_blocking=True is only meaningful for CPU->CUDA
        transfers; it's pure overhead (and wasted pinned-pool pages) for
        CPU->CPU. Gate it on the actual target device type.
        """

        target_is_cuda = self.device.type == "cuda"
        moved_batch: Dict[str, torch.Tensor] = {}

        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                if target_is_cuda and value.device.type == "cpu" and not value.is_pinned():
                    try:
                        value = value.pin_memory()
                    except RuntimeError:
                        # pin_memory can fail under pressure; fall back silently.
                        pass
                moved_batch[key] = value.to(
                    self.device,
                    non_blocking=target_is_cuda,
                )
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
            if self.return_logits_only:
                return outputs

            softmax = torch.softmax
            argmax = torch.argmax
            results: Dict[str, Any] = {}

            for key, value in outputs.items():
                if isinstance(value, torch.Tensor):
                    # Match only real logits keys, not e.g. "logits_norm" or
                    # "per_sample_logits_aux". Exact equality or an "_logits"
                    # suffix is what our heads actually produce.
                    is_logits = key == "logits" or key.endswith("_logits")
                    if is_logits:
                        logits = value
                        if self.compute_probabilities:
                            probs = softmax(logits, dim=-1)
                            calibrated_probs = self._calibrate_probabilities(
                                logits=logits,
                                probabilities=probs,
                            )
                        else:
                            probs = None
                            calibrated_probs = None

                        preds = argmax(logits, dim=-1)
                        # Strip the suffix precisely; avoids mangling keys
                        # where "logits" appears as a substring elsewhere.
                        if key == "logits":
                            base = ""
                        else:
                            base = key[: -len("_logits")] + "_"
                        results[f"{base}predictions"] = preds

                        if probs is not None:
                            results[f"{base}probabilities"] = probs
                        if calibrated_probs is not None:
                            results[f"{base}calibrated_probabilities"] = calibrated_probs
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
                logits_device = logits
                calibrated = self.temperature_scaler.predict_proba(logits_device)
                return calibrated.to(probabilities.device)
            except Exception as exc:  # noqa: BLE001
                logger.warning("Temperature scaling skipped: %s", exc)

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
                logger.warning("Isotonic calibration skipped: %s", exc)

        return probabilities

    def _run_ensemble(self, batch: Dict[str, torch.Tensor]) -> Dict[str, Any]:
        if self.ensemble_model is None:
            raise RuntimeError("Ensemble model is not configured.")

        logits: Optional[torch.Tensor] = None
        outputs = self.ensemble_model(**batch)

        if isinstance(outputs, torch.Tensor):
            logits = outputs
        elif isinstance(outputs, dict):
            for key, value in outputs.items():
                if isinstance(value, torch.Tensor) and "logits" in key:
                    logits = value
                    break
            if logits is None:
                for value in outputs.values():
                    if isinstance(value, torch.Tensor):
                        logits = value
                        break

        if not isinstance(logits, torch.Tensor):
            raise RuntimeError("Ensemble model must return logits tensor.")

        return {"ensemble_logits": logits}
