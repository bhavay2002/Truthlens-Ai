from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

from src.explainability.utils_validation import validate_tokens_scores

logger = logging.getLogger(__name__)


@dataclass
class AggregationWeights:
    shap: float = 0.4
    integrated_gradients: float = 0.3
    attention: float = 0.2
    lime: float = 0.1


@dataclass
class AggregatedExplanation:
    tokens: List[str]
    final_token_importance: List[float]
    confidence_score: float
    agreement_score: float


class ExplanationAggregator:
    def __init__(self, weights: Optional[AggregationWeights] = None) -> None:
        w = weights or AggregationWeights()
        total = w.shap + w.integrated_gradients + w.attention + w.lime
        if total <= 0:
            raise ValueError("Aggregation weights must sum to a positive value.")
        self.weights = AggregationWeights(
            shap=w.shap / total,
            integrated_gradients=w.integrated_gradients / total,
            attention=w.attention / total,
            lime=w.lime / total,
        )

    @staticmethod
    def _normalize(v: np.ndarray) -> np.ndarray:
        v = np.abs(np.asarray(v, dtype=float))
        s = float(np.sum(v))
        if s <= 0:
            return np.zeros_like(v)
        return v / s

    @staticmethod
    def _as_map(items: List[Dict], key: str) -> Dict[str, float]:
        out: Dict[str, float] = {}
        for it in items:
            tok = str(it.get("token"))
            out[tok] = float(it.get(key, 0.0))
        return out

    @staticmethod
    def _lime_map(items: List[Tuple[str, float]]) -> Dict[str, float]:
        return {str(t): float(s) for t, s in items}

    def aggregate(
        self,
        shap_importance: Optional[List[Dict]] = None,
        integrated_gradients: Optional[List[Dict]] = None,
        attention_scores: Optional[List[Dict]] = None,
        lime_importance: Optional[List] = None,
    ) -> Dict:
        sources: List[Tuple[Dict[str, float], float]] = []

        if shap_importance:
            sources.append((self._as_map(shap_importance, "importance"), self.weights.shap))
        if integrated_gradients:
            sources.append((self._as_map(integrated_gradients, "importance"), self.weights.integrated_gradients))
        if attention_scores:
            sources.append((self._as_map(attention_scores, "attention"), self.weights.attention))
        if lime_importance:
            sources.append((self._lime_map(lime_importance), self.weights.lime))

        if not sources:
            raise ValueError("No valid explanation sources provided.")

        common = set(sources[0][0].keys())
        for m, _ in sources[1:]:
            common &= set(m.keys())
        if not common:
            raise ValueError("No common tokens across explanation methods.")

        tokens = sorted(common)
        weighted_rows = []
        for m, w in sources:
            vec = np.array([m[t] for t in tokens], dtype=float)
            weighted_rows.append(self._normalize(vec) * w)

        matrix = np.vstack(weighted_rows)
        final_scores = self._normalize(matrix.sum(axis=0))

        validate_tokens_scores(tokens, final_scores.tolist())

        corrs = []
        if matrix.shape[0] > 1:
            for i in range(matrix.shape[0]):
                for j in range(i + 1, matrix.shape[0]):
                    c = np.corrcoef(matrix[i], matrix[j])[0, 1]
                    if not np.isnan(c):
                        corrs.append(c)
        agreement = float(np.mean(corrs)) if corrs else (1.0 if matrix.shape[0] == 1 else 0.0)
        confidence = float(np.mean(np.max(matrix, axis=1)))

        return AggregatedExplanation(
            tokens=tokens,
            final_token_importance=final_scores.tolist(),
            confidence_score=confidence,
            agreement_score=agreement,
        ).__dict__
