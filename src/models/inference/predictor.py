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
from typing import Dict, Any

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class Predictor:
    """
    Generic predictor wrapper for TruthLens models.
    """

    def __init__(
        self,
        model: nn.Module,
        device: str | None = None,
    ) -> None:

        if not isinstance(model, nn.Module):
            raise TypeError("model must be torch.nn.Module")

        self.model = model

        self.device = torch.device(
            device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        )

        self.model.to(self.device)
        self.model.eval()

        logger.info("Predictor initialized on device %s", self.device)

    @torch.no_grad()
    def predict_batch(
        self,
        batch: Dict[str, torch.Tensor],
    ) -> Dict[str, Any]:
        """
        Run prediction for a batch of inputs.
        """

        batch = self._move_to_device(batch)

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
                        preds = torch.argmax(probs, dim=-1)

                        formatted[key.replace("logits", "predictions")] = preds
                        formatted[key.replace("logits", "probabilities")] = probs

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
