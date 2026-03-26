"""
File Name: propaganda_explainer.py
Module: Model Analysis - Propaganda Explanation
Description:
    Provides interpretability utilities for propaganda detection outputs in the
    TruthLens AI system. The module extracts token-level importance signals from
    transformer models using attention aggregation and gradient-based scoring.
    It produces human-readable explanations showing which parts of the text
    contributed most to propaganda predictions.

Dependencies:
    logging
    typing
    torch
    numpy
    transformers

Inputs:
    Tokenized model inputs
    Trained propaganda detection model
    Token list

Outputs:
    Token importance scores and explanation dictionary
"""

import logging
from typing import Dict, List

import numpy as np
import torch


logger = logging.getLogger(__name__)


class PropagandaExplainer:
    """
    Generates token-level explanations for propaganda detection predictions.
    """

    def __init__(self, model) -> None:
        """Initialize explainer with trained model."""

        if model is None:
            raise ValueError("model cannot be None")

        self.model = model

        logger.info("PropagandaExplainer initialized")

    def _resolve_model_device(self) -> torch.device | None:
        try:
            return next(self.model.parameters()).device
        except (AttributeError, StopIteration, TypeError):
            return None

    def explain(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        tokens: List[str],
    ) -> Dict[str, float]:
        """Generate token-level importance scores."""

        if input_ids is None or attention_mask is None:
            raise ValueError("input tensors cannot be None")
        if not isinstance(input_ids, torch.Tensor) or not isinstance(
            attention_mask, torch.Tensor
        ):
            raise TypeError("input_ids and attention_mask must be torch tensors")
        if input_ids.ndim != 2 or attention_mask.ndim != 2:
            raise ValueError("input_ids and attention_mask must be 2D tensors")
        if input_ids.shape != attention_mask.shape:
            raise ValueError(
                "input_ids and attention_mask must have identical shapes"
            )
        if not isinstance(tokens, list) or not tokens:
            raise ValueError("tokens must be a non-empty list")

        sequence_length = input_ids.shape[1]
        if len(tokens) != sequence_length:
            raise ValueError(
                "tokens length must match model sequence length "
                f"({len(tokens)} != {sequence_length})."
            )

        model_device = self._resolve_model_device()
        if model_device is not None:
            input_ids = input_ids.to(model_device)
            attention_mask = attention_mask.to(model_device)

        try:
            if hasattr(self.model, "zero_grad"):
                self.model.zero_grad(set_to_none=True)

            embedding_layer = self.model.get_input_embeddings()
            input_embeddings = (
                embedding_layer(input_ids).detach().requires_grad_(True)
            )

            model_kwargs = {
                "inputs_embeds": input_embeddings,
                "attention_mask": attention_mask,
            }
            outputs = self.model(**model_kwargs)

            logits = getattr(outputs, "logits", None)
            if logits is None and isinstance(outputs, dict):
                logits = outputs.get("logits")
            if logits is None:
                raise RuntimeError(
                    "Model output does not contain logits for explanation."
                )

            score = logits.max()

            score.backward()

        except Exception as exc:
            logger.exception("Explanation computation failed")
            raise RuntimeError("Propaganda explanation failed") from exc

        gradients = input_embeddings.grad

        if gradients is None:
            raise RuntimeError("Gradients could not be computed")

        importance_scores = (
            torch.abs(gradients)
            .sum(dim=-1)
            .detach()
            .cpu()
            .numpy()
        )

        token_scores = self._normalize_scores(importance_scores[0], tokens)

        explanation: dict[str, float] = {}
        token_occurrences: dict[str, int] = {}
        for token, score in zip(tokens, token_scores):
            count = token_occurrences.get(token, 0)
            key = token if count == 0 else f"{token}_{count}"
            explanation[key] = float(score)
            token_occurrences[token] = count + 1

        return explanation

    def _normalize_scores(
        self,
        scores: np.ndarray,
        tokens: List[str],
    ) -> np.ndarray:
        """Normalize token importance scores."""

        if scores.size == 0:
            return np.zeros(len(tokens))

        scores = np.asarray(scores[: len(tokens)], dtype=float)
        scores = np.abs(scores)

        total = float(np.sum(scores))

        if total <= 0:
            return np.zeros(len(tokens), dtype=float)

        normalized = scores / total

        return normalized
