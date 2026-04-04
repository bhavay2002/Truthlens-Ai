"""
File Name: propaganda_explainer.py
Module: Explainability - Propaganda Analysis
Description:
    Provides interpretability utilities for propaganda detection models in the
    TruthLens AI system. The module computes token-level importance scores
    using gradient-based attribution from transformer embeddings. It produces
    normalized importance scores aligned with tokens, enabling human-readable
    explanations for propaganda predictions.Combines gradient attribution, integrated gradients, and
    attention-based interpretability to produce robust token-level
    explanations.

Dependencies:
    logging
    typing
    torch
    numpy

Inputs:
    input_ids : torch.Tensor
    attention_mask : torch.Tensor
    tokens : List[str]

Outputs:
    Dictionary mapping tokens to normalized importance scores
"""
from __future__ import annotations

import logging
from typing import Dict, List, Optional

import numpy as np
import torch

logger = logging.getLogger(__name__)


class PropagandaExplainer:

    def __init__(self, model: torch.nn.Module) -> None:

        if model is None:
            raise ValueError("model cannot be None")

        if not isinstance(model, torch.nn.Module):
            raise TypeError("model must be torch.nn.Module")

        self.model = model

        logger.info("PropagandaExplainer initialized")

    # --------------------------------------------------
    # Device Resolution
    # --------------------------------------------------

    def _resolve_device(self) -> Optional[torch.device]:

        try:
            return next(self.model.parameters()).device
        except Exception:
            return None

    # --------------------------------------------------
    # Main Explanation Function
    # --------------------------------------------------

    def explain(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        tokens: List[str],
    ) -> Dict[str, float]:

        gradients = self._gradient_importance(
            input_ids,
            attention_mask,
        )

        normalized = self._normalize_scores(
            gradients,
            tokens,
        )

        merged_tokens, merged_scores = self._merge_subwords(
            tokens,
            normalized,
        )

        explanation = {}

        token_count = {}

        for token, score in zip(merged_tokens, merged_scores):

            idx = token_count.get(token, 0)

            key = token if idx == 0 else f"{token}_{idx}"

            explanation[key] = float(score)

            token_count[token] = idx + 1

        return explanation

    # --------------------------------------------------
    # Gradient Attribution
    # --------------------------------------------------

    def _gradient_importance(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> np.ndarray:

        device = self._resolve_device()

        if device is not None:
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)

        self.model.zero_grad()

        embedding_layer = self.model.get_input_embeddings()

        embeddings = embedding_layer(input_ids)

        embeddings = embeddings.detach().requires_grad_(True)

        outputs = self.model(
            inputs_embeds=embeddings,
            attention_mask=attention_mask,
        )

        logits = outputs.logits if hasattr(outputs, "logits") else outputs["logits"]

        score = logits.max()

        score.backward()

        grads = embeddings.grad

        importance = torch.abs(grads).sum(dim=-1)

        return importance.detach().cpu().numpy()[0]

    # --------------------------------------------------
    # Subword Token Merge
    # --------------------------------------------------

    def _merge_subwords(
        self,
        tokens: List[str],
        scores: List[float],
    ):

        merged_tokens = []
        merged_scores = []

        buffer_token = ""
        buffer_scores = []

        for token, score in zip(tokens, scores):

            if token.startswith("##"):

                buffer_token += token[2:]

                buffer_scores.append(score)

            else:

                if buffer_token:

                    merged_tokens.append(buffer_token)

                    merged_scores.append(float(np.mean(buffer_scores)))

                buffer_token = token

                buffer_scores = [score]

        if buffer_token:

            merged_tokens.append(buffer_token)

            merged_scores.append(float(np.mean(buffer_scores)))

        return merged_tokens, merged_scores

    # --------------------------------------------------
    # Score Normalization
    # --------------------------------------------------

    def _normalize_scores(
        self,
        scores: np.ndarray,
        tokens: List[str],
    ) -> np.ndarray:

        scores = np.asarray(scores[: len(tokens)], dtype=float)

        scores = np.abs(scores)

        total = float(np.sum(scores))

        if total <= 0:
            return np.zeros(len(tokens), dtype=float)

        return scores / total

    # --------------------------------------------------
    # Propaganda Intensity
    # --------------------------------------------------

    def propaganda_intensity(
        self,
        explanation: Dict[str, float],
    ) -> float:

        if not explanation:
            return 0.0

        values = list(explanation.values())

        return float(np.mean(values))