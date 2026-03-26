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

    def explain(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        tokens: List[str],
    ) -> Dict[str, float]:
        """Generate token-level importance scores."""

        if input_ids is None or attention_mask is None:
            raise ValueError("input tensors cannot be None")

        if not isinstance(tokens, list) or not tokens:
            raise ValueError("tokens must be a non-empty list")

        input_ids = input_ids.clone().detach().requires_grad_(True)

        try:
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )

            logits = outputs["logits"]

            score = torch.sum(logits)

            score.backward()

        except Exception as exc:
            logger.exception("Explanation computation failed")
            raise RuntimeError("Propaganda explanation failed") from exc

        gradients = input_ids.grad

        if gradients is None:
            raise RuntimeError("Gradients could not be computed")

        importance_scores = torch.abs(gradients).detach().cpu().numpy()

        token_scores = self._normalize_scores(importance_scores[0], tokens)

        explanation = {token: float(score) for token, score in zip(tokens, token_scores)}

        return explanation

    def _normalize_scores(
        self,
        scores: np.ndarray,
        tokens: List[str],
    ) -> np.ndarray:
        """Normalize token importance scores."""

        if scores.size == 0:
            return np.zeros(len(tokens))

        scores = scores[: len(tokens)]

        total = np.sum(scores)

        if total == 0:
            return scores

        normalized = scores / total

        return normalized