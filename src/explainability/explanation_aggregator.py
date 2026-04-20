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
            tokens = [e["token"] for e in explanations if isinstance(e, dict) and "token" in e and key in e]
            scores = [e[key] for e in explanations if isinstance(e, dict) and "token" in e and key in e]

            if not tokens or len(tokens) != len(scores):
                return explanations

            validate_tokens_scores(tokens, scores)
            tokens, scores = align_tokens(tokens, scores)

            return [{"token": t, key: float(s)} for t, s in zip(tokens, scores)]

        def _to_token_map(explanations: List[Dict], key: str) -> Dict[str, float]:
            return {e["token"]: float(e[key]) for e in explanations}

        def aggregate_sources(
            sources: List[Tuple[List[Dict], str, float]],
        ) -> Dict[str, float]:
            aligned: List[Tuple[Dict[str, float], float]] = []
            for exp, key, weight in sources:
                exp = _align_input(exp, key)
                aligned.append((_to_token_map(exp, key), weight))

            token_sets = [set(m.keys()) for m, _ in aligned]
            if not token_sets:
                return {}
            tokens = sorted(set.union(*token_sets))

            final: Dict[str, float] = {}
            for t in tokens:
                score = 0.0
                total_weight = 0.0
                for m, w in aligned:
                    if t in m:
                        score += w * m[t]
                        total_weight += w
                if total_weight > 0:
                    final[t] = score / total_weight

            return final

        sources: List[Tuple[List[Dict], str, float]] = []

        if shap_importance:
            sources.append((shap_importance, "importance", self.weights.shap))
        if integrated_gradients:
            sources.append((integrated_gradients, "importance", self.weights.integrated_gradients))
        if attention_scores:
            sources.append((attention_scores, "attention", self.weights.attention))

        if not sources and lime_importance:
            # LIME uses (token, score) tuples; keep as separate mapping.
            lime_map = self._lime_map(lime_importance)
            final_map = lime_map
        else:
            final_map = aggregate_sources(sources)

        if lime_importance and sources:
            lime_map = self._lime_map(lime_importance)
            for token, score in lime_map.items():
                if token in final_map:
                    continue
                final_map[token] = score

        if not final_map:
            raise ValueError("No valid explanation sources provided.")

        tokens = sorted(final_map.keys())
        final_scores = np.array([final_map[t] for t in tokens], dtype=float)
        final_scores = self._normalize(final_scores)

        validate_tokens_scores(tokens, final_scores.tolist())

        confidence = float(np.mean(np.abs(final_scores))) if final_scores.size else 0.0

        return AggregatedExplanation(
            tokens=tokens,
            final_token_importance=final_scores.tolist(),
            confidence_score=confidence,
            agreement_score=0.0,
        ).__dict__
