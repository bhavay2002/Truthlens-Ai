from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

from src.explainability.token_alignment import align_tokens
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

    @staticmethod
    def _safe_corr(a: np.ndarray, b: np.ndarray) -> float:
        if a.size < 2 or b.size < 2:
            return 0.0
        if np.std(a) < 1e-12 or np.std(b) < 1e-12:
            return 0.0
        c = np.corrcoef(a, b)[0, 1]
        return 0.0 if np.isnan(c) else float(c)

    def aggregate(
        self,
        shap_importance: Optional[List[Dict]] = None,
        integrated_gradients: Optional[List[Dict]] = None,
        attention_scores: Optional[List[Dict]] = None,
        lime_importance: Optional[List] = None,
    ) -> Dict:
        def _align_input(explanations: List[Dict], key: str) -> List[Dict]:
            tokens = []
            scores = []
            for item in explanations:
                if not isinstance(item, dict):
                    continue
                token = item.get("token")
                score = item.get(key)
                if isinstance(token, str) and isinstance(score, (int, float)):
                    tokens.append(token)
                    scores.append(score)
            if not tokens or len(tokens) != len(scores):
                return explanations
            validate_tokens_scores(tokens, scores)
            tokens, scores = align_tokens(tokens, scores)
            return [{"token": t, key: float(s)} for t, s in zip(tokens, scores)]

        def _align_lime(items: List) -> List[Tuple[str, float]]:
            tokens = []
            scores = []
            for item in items:
                if not isinstance(item, (list, tuple)) or len(item) < 2:
                    continue
                token, score = item[0], item[1]
                if isinstance(token, str) and isinstance(score, (int, float)):
                    tokens.append(token)
                    scores.append(score)
            if not tokens or len(tokens) != len(scores):
                return items
            validate_tokens_scores(tokens, scores)
            tokens, scores = align_tokens(tokens, scores)
            return list(zip(tokens, scores))

        if shap_importance:
            shap_importance = _align_input(shap_importance, "importance")
        if integrated_gradients:
            integrated_gradients = _align_input(integrated_gradients, "importance")
        if attention_scores:
            attention_scores = _align_input(attention_scores, "attention")
        if lime_importance:
            lime_importance = _align_lime(lime_importance)

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

        all_tokens = set()
        for m, _ in sources:
            all_tokens.update(m.keys())
        if not all_tokens:
            raise ValueError("No tokens found across explanation methods.")

        tokens = sorted(all_tokens)
        weighted_rows = []
        for m, w in sources:
            vec = np.array([m.get(t, 0.0) for t in tokens], dtype=float)
            present_values = np.array([m[t] for t in tokens if t in m], dtype=float)

            if len(present_values) > 0:
                norm = np.sum(np.abs(present_values))
                if norm > 0:
                    vec = vec / norm

            weighted_rows.append(vec * w)

        matrix = np.vstack(weighted_rows)
        final_scores = self._normalize(matrix.sum(axis=0))

        validate_tokens_scores(tokens, final_scores.tolist())

        corrs = []
        if matrix.shape[0] > 1:
            for i in range(matrix.shape[0]):
                for j in range(i + 1, matrix.shape[0]):
                    corrs.append(self._safe_corr(matrix[i], matrix[j]))
        agreement = float(np.mean(corrs)) if corrs else (1.0 if matrix.shape[0] == 1 else 0.0)
        confidence = float(np.mean(np.max(matrix, axis=1)))

        return AggregatedExplanation(
            tokens=tokens,
            final_token_importance=final_scores.tolist(),
            confidence_score=confidence,
            agreement_score=agreement,
        ).__dict__
