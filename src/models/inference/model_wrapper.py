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

import torch
import torch.nn as nn


logger = logging.getLogger(__name__)


class ModelWrapper:
    """
    Wrapper for managing model lifecycle and inference execution.
    """

    def __init__(
        self,
        model: nn.Module,
        device: Optional[str] = None,
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

        logger.info("ModelWrapper initialized on device: %s", self.device)

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
                        preds = torch.argmax(probs, dim=-1)

                        results[key.replace("logits", "probabilities")] = probs
                        results[key.replace("logits", "predictions")] = preds

                    else:
                        results[key] = value

                else:
                    results[key] = value

            return results

        raise RuntimeError("Unsupported model output format")