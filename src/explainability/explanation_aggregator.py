"""
File Name: explanation_aggregator.py
Module: Explainability - Aggregation
Description:
    Aggregates multiple explanation methods into a unified token importance
    representation for TruthLens AI.

    Supported sources:
        • SHAP explanations
        • Integrated Gradients
        • Attention scores
        • LIME explanations

    The module combines these signals using configurable weights and produces:

        - final_token_importance
        - confidence_score
        - agreement_score

    This aggregation helps create a single interpretable explanation signal
    useful for dashboards, reports, and research evaluation.

Dependencies:
    logging
    dataclasses
    typing
    numpy

Inputs:
    shap_importance
    integrated_gradients
    attention_scores
    lime_importance

Outputs:
    Aggregated explanation dictionary
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class AggregationWeights:
    """Weights used for combining explanation methods."""

    shap: float = 0.4
    integrated_gradients: float = 0.3
    attention: float = 0.2
    lime: float = 0.1


@dataclass
class AggregatedExplanation:
    """Structured aggregated explanation result."""

    tokens: List[str]
    final_token_importance: List[float]
    confidence_score: float
    agreement_score: float


class ExplanationAggregator:
    """
    Aggregates explanations from multiple attribution methods.
    """

    def __init__(self, weights: Optional[AggregationWeights] = None) -> None:
        self.weights = weights or AggregationWeights()

        total = (
            self.weights.shap
            + self.weights.integrated_gradients
            + self.weights.attention
            + self.weights.lime
        )

        if total <= 0:
            raise ValueError("Aggregation weights must sum to a positive value.")

        logger.info("ExplanationAggregator initialized")

    @staticmethod
    def _tokens_and_scores(items: List[Dict], score_key: str):
        tokens = []
        scores = []

        for item in items:
            token = str(item.get("token"))
            score = float(item.get(score_key, 0.0))
            tokens.append(token)
            scores.append(score)

        return tokens, np.asarray(scores, dtype=float)

    @staticmethod
    def _normalize(scores: np.ndarray) -> np.ndarray:
        scores = np.asarray(scores, dtype=float)

        if scores.size == 0:
            return scores

        scores = np.abs(scores)

        total = float(np.sum(scores))

        if total <= 0:
            return np.zeros_like(scores)

        return scores / total

    @staticmethod
    def _compute_agreement(matrix: np.ndarray) -> float:
        """
        Compute agreement between explanation methods using average
        pairwise correlation.
        """

        if matrix.shape[0] < 2:
            return 1.0

        correlations = []

        for i in range(matrix.shape[0]):
            for j in range(i + 1, matrix.shape[0]):
                corr = np.corrcoef(matrix[i], matrix[j])[0, 1]

                if not np.isnan(corr):
                    correlations.append(corr)

        if not correlations:
            return 0.0

        return float(np.mean(correlations))

    def aggregate(
        self,
        shap_importance: Optional[List[Dict]] = None,
        integrated_gradients: Optional[List[Dict]] = None,
        attention_scores: Optional[List[Dict]] = None,
        lime_importance: Optional[List] = None,
    ) -> Dict:
        """
        Aggregate multiple explanation sources.
        """

        methods = []

        tokens = None

        if shap_importance:
            tokens, scores = self._tokens_and_scores(shap_importance, "importance")
            methods.append(self._normalize(scores) * self.weights.shap)

        if integrated_gradients:
            tokens, scores = self._tokens_and_scores(
                integrated_gradients, "importance"
            )
            methods.append(
                self._normalize(scores) * self.weights.integrated_gradients
            )

        if attention_scores:
            tokens, scores = self._tokens_and_scores(attention_scores, "attention")
            methods.append(self._normalize(scores) * self.weights.attention)

        if lime_importance:
            tokens = [token for token, _ in lime_importance]
            scores = np.array([float(score) for _, score in lime_importance])
            methods.append(self._normalize(scores) * self.weights.lime)

        if not methods or tokens is None:
            raise ValueError("No valid explanation sources provided.")

        min_length = min(len(m) for m in methods)
        methods = [m[:min_length] for m in methods]
        tokens = list(tokens)[:min_length]

        matrix = np.vstack(methods)

        final_scores = matrix.sum(axis=0)

        final_scores = self._normalize(final_scores)

        agreement_score = self._compute_agreement(matrix)

        confidence_score = float(np.mean(np.max(matrix, axis=1)))

        explanation = AggregatedExplanation(
            tokens=tokens,
            final_token_importance=final_scores.tolist(),
            confidence_score=confidence_score,
            agreement_score=agreement_score,
        )

        return explanation.__dict__