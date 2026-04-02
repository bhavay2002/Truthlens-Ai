"""
File Name: explanation_metrics.py
Module: Explainability - Evaluation Metrics
Description:
    Provides quantitative evaluation metrics for explanation methods used
    in the TruthLens AI system. These metrics measure how well explanation
    signals reflect the model's true decision-making behavior.

    Implemented metrics commonly used in interpretability research:

        • Faithfulness
        • Comprehensiveness
        • Sufficiency
        • Deletion score
        • Insertion score

    These metrics evaluate whether removing or inserting tokens identified
    as important changes the model prediction in a meaningful way.

Dependencies:
    logging
    typing
    numpy

Inputs:
    text
    tokens
    importance scores
    prediction function

Outputs:
    dictionary of explanation evaluation metrics
"""

from __future__ import annotations

import logging
from typing import Callable, Dict, List

import numpy as np

logger = logging.getLogger(__name__)

PredictionFn = Callable[[str], Dict[str, float]]


class ExplanationMetrics:
    """
    Compute evaluation metrics for explanation methods.
    """

    def __init__(self) -> None:
        logger.info("ExplanationMetrics initialized")

    @staticmethod
    def _extract_fake_probability(result: Dict[str, float]) -> float:
        if "fake_probability" not in result:
            raise KeyError("Prediction output must contain 'fake_probability'")
        return float(result["fake_probability"])

    @staticmethod
    def _remove_tokens(tokens: List[str], indices: List[int]) -> str:
        remaining = [t for i, t in enumerate(tokens) if i not in indices]
        return " ".join(remaining)

    @staticmethod
    def _keep_tokens(tokens: List[str], indices: List[int]) -> str:
        kept = [tokens[i] for i in indices if i < len(tokens)]
        return " ".join(kept)

    @staticmethod
    def _sort_indices(scores: List[float]) -> List[int]:
        scores_arr = np.asarray(scores)
        return list(np.argsort(scores_arr)[::-1])

    def faithfulness(
        self,
        tokens: List[str],
        scores: List[float],
        predict_fn: PredictionFn,
    ) -> float:
        """
        Measure correlation between token importance and prediction change.
        """
        base_pred = self._extract_fake_probability(
            predict_fn(" ".join(tokens))
        )

        deltas = []

        for i, token in enumerate(tokens):
            perturbed_text = self._remove_tokens(tokens, [i])
            perturbed_pred = self._extract_fake_probability(
                predict_fn(perturbed_text)
            )
            delta = base_pred - perturbed_pred
            deltas.append(delta)

        if len(deltas) < 2:
            return 0.0

        corr = np.corrcoef(scores, deltas)[0, 1]

        if np.isnan(corr):
            return 0.0

        return float(corr)

    def comprehensiveness(
        self,
        tokens: List[str],
        scores: List[float],
        predict_fn: PredictionFn,
        top_k: int = 5,
    ) -> float:
        """
        Remove top-k important tokens and measure prediction drop.
        """
        base_pred = self._extract_fake_probability(
            predict_fn(" ".join(tokens))
        )

        ranked = self._sort_indices(scores)[:top_k]

        perturbed_text = self._remove_tokens(tokens, ranked)

        new_pred = self._extract_fake_probability(predict_fn(perturbed_text))

        return float(base_pred - new_pred)

    def sufficiency(
        self,
        tokens: List[str],
        scores: List[float],
        predict_fn: PredictionFn,
        top_k: int = 5,
    ) -> float:
        """
        Keep only top-k tokens and evaluate prediction preservation.
        """
        base_pred = self._extract_fake_probability(
            predict_fn(" ".join(tokens))
        )

        ranked = self._sort_indices(scores)[:top_k]

        reduced_text = self._keep_tokens(tokens, ranked)

        new_pred = self._extract_fake_probability(predict_fn(reduced_text))

        return float(base_pred - new_pred)

    def deletion_score(
        self,
        tokens: List[str],
        scores: List[float],
        predict_fn: PredictionFn,
    ) -> float:
        """
        Gradually remove tokens by importance and compute prediction drop area.
        """
        ranked = self._sort_indices(scores)

        base_pred = self._extract_fake_probability(
            predict_fn(" ".join(tokens))
        )

        preds = []

        current_tokens = tokens.copy()

        for idx in ranked:
            if idx < len(current_tokens):
                current_tokens[idx] = ""

            text = " ".join([t for t in current_tokens if t])
            pred = self._extract_fake_probability(predict_fn(text))
            preds.append(pred)

        preds = np.asarray(preds)

        return float(base_pred - preds.mean())

    def insertion_score(
        self,
        tokens: List[str],
        scores: List[float],
        predict_fn: PredictionFn,
    ) -> float:
        """
        Gradually insert important tokens and measure prediction increase.
        """
        ranked = self._sort_indices(scores)

        preds = []

        inserted_tokens: List[str] = []

        for idx in ranked:
            if idx < len(tokens):
                inserted_tokens.append(tokens[idx])

            text = " ".join(inserted_tokens)
            pred = self._extract_fake_probability(predict_fn(text))
            preds.append(pred)

        preds = np.asarray(preds)

        return float(preds.mean())

    def evaluate(
        self,
        tokens: List[str],
        scores: List[float],
        predict_fn: PredictionFn,
    ) -> Dict[str, float]:
        """
        Compute all explanation evaluation metrics.
        """

        results = {
            "faithfulness": self.faithfulness(tokens, scores, predict_fn),
            "comprehensiveness": self.comprehensiveness(
                tokens, scores, predict_fn
            ),
            "sufficiency": self.sufficiency(tokens, scores, predict_fn),
            "deletion_score": self.deletion_score(
                tokens, scores, predict_fn
            ),
            "insertion_score": self.insertion_score(
                tokens, scores, predict_fn
            ),
        }

        logger.info("Explanation evaluation metrics computed")

        return results