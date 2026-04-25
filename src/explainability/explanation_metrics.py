from __future__ import annotations

import logging
from typing import Callable, Dict, List, Optional

import numpy as np

from src.explainability.utils_validation import validate_tokens_scores

logger = logging.getLogger(__name__)
EPS = 1e-12

PredictionFn = Callable[[List[str]], List[Dict[str, float]]]


class ExplanationMetrics:

    def __init__(self) -> None:
        logger.info("ExplanationMetrics initialized")

    # =====================================================
    # UTILS
    # =====================================================

    @staticmethod
    def _extract_fake_prob_batch(results):
        return np.array([r["fake_probability"] for r in results], dtype=float)

    @staticmethod
    def _sort_indices(scores):
        return list(np.argsort(np.asarray(scores))[::-1])

    @staticmethod
    def _normalize(x):
        x = np.asarray(x, dtype=float)

        if x.size == 0:
            return x

        mn, mx = np.min(x), np.max(x)
        if mx - mn < EPS:
            return np.zeros_like(x)

        return (x - mn) / (mx - mn + EPS)

    @staticmethod
    def _apply_confidence(value: float, confidence: Optional[float]) -> float:
        if confidence is None:
            return float(value)
        return float(value * np.clip(confidence, 0.0, 1.0))

    # =====================================================
    # FAITHFULNESS (BATCHED)
    # =====================================================

    def faithfulness(self, tokens, scores, predict_fn):

        validate_tokens_scores(tokens, scores)

        base = predict_fn([" ".join(tokens)])[0]["fake_probability"]

        texts = [
            " ".join([t for j, t in enumerate(tokens) if j != i])
            for i in range(len(tokens))
        ]

        preds = self._extract_fake_prob_batch(predict_fn(texts))
        deltas = base - preds

        if len(deltas) < 2:
            return 0.0

        corr = np.corrcoef(scores, deltas)[0, 1]
        return 0.0 if np.isnan(corr) else float(corr)

    # =====================================================
    # COMPREHENSIVENESS
    # =====================================================

    def comprehensiveness(self, tokens, scores, predict_fn, top_k=5):

        base = predict_fn([" ".join(tokens)])[0]["fake_probability"]

        ranked = self._sort_indices(scores)[:top_k]

        perturbed = " ".join([t for i, t in enumerate(tokens) if i not in ranked])
        new = predict_fn([perturbed])[0]["fake_probability"]

        return float(base - new)

    # =====================================================
    # SUFFICIENCY
    # =====================================================

    def sufficiency(self, tokens, scores, predict_fn, top_k=5):

        base = predict_fn([" ".join(tokens)])[0]["fake_probability"]

        ranked = self._sort_indices(scores)[:top_k]
        kept = [tokens[i] for i in ranked]

        new = predict_fn([" ".join(kept)])[0]["fake_probability"]

        return float(base - new)

    # =====================================================
    # DELETION (BATCHED)
    # =====================================================

    def deletion_score(self, tokens, scores, predict_fn):

        ranked = self._sort_indices(scores)

        texts = []
        current = tokens.copy()

        for idx in ranked:
            current[idx] = ""
            texts.append(" ".join([t for t in current if t]))

        preds = self._extract_fake_prob_batch(predict_fn(texts))

        base = preds[0] if len(preds) > 0 else 0.0
        return float(base - np.mean(preds))

    # =====================================================
    # INSERTION (BATCHED)
    # =====================================================

    def insertion_score(self, tokens, scores, predict_fn):

        ranked = self._sort_indices(scores)

        slots = [""] * len(tokens)
        texts = []

        for idx in ranked:
            slots[idx] = tokens[idx]
            texts.append(" ".join([t for t in slots if t]))

        preds = self._extract_fake_prob_batch(predict_fn(texts))

        return float(np.trapz(preds))

    # =====================================================
    # VARIANCE (🔥 NEW)
    # =====================================================

    def variance(self, scores: List[float]) -> float:
        arr = np.asarray(scores, dtype=float)
        if arr.size == 0:
            return 0.0
        return float(np.var(arr))

    # =====================================================
    # SINGLE EVALUATION
    # =====================================================

    def evaluate(
        self,
        tokens,
        scores,
        predict_fn,
        *,
        confidence: Optional[float] = None,
    ) -> Dict[str, float]:

        validate_tokens_scores(tokens, scores)

        raw = {
            "faithfulness": self.faithfulness(tokens, scores, predict_fn),
            "comprehensiveness": self.comprehensiveness(tokens, scores, predict_fn),
            "sufficiency": self.sufficiency(tokens, scores, predict_fn),
            "deletion_score": self.deletion_score(tokens, scores, predict_fn),
            "insertion_score": self.insertion_score(tokens, scores, predict_fn),
        }

        # 🔥 confidence weighting
        weighted = {
            k: self._apply_confidence(v, confidence)
            for k, v in raw.items()
        }

        values = list(weighted.values())
        norm = self._normalize(values)

        result = {
            **weighted,
            "variance": self.variance(scores),
            "normalized": dict(zip(weighted.keys(), norm.tolist())),
            "overall_score": float(np.mean(norm)) if len(norm) > 0 else 0.0,
        }

        return result

    # =====================================================
    # BATCH EVALUATION (🔥 NEW)
    # =====================================================

    def evaluate_batch(
        self,
        batch_tokens: List[List[str]],
        batch_scores: List[List[float]],
        predict_fn: PredictionFn,
        *,
        confidences: Optional[List[float]] = None,
    ) -> Dict[str, float]:

        results = []

        for i, (tokens, scores) in enumerate(zip(batch_tokens, batch_scores)):

            conf = confidences[i] if confidences and i < len(confidences) else None

            res = self.evaluate(
                tokens,
                scores,
                predict_fn,
                confidence=conf,
            )

            results.append(res["overall_score"])

        arr = np.asarray(results, dtype=float)

        return {
            "batch_mean": float(np.mean(arr)) if arr.size else 0.0,
            "batch_std": float(np.std(arr)) if arr.size else 0.0,
            "batch_min": float(np.min(arr)) if arr.size else 0.0,
            "batch_max": float(np.max(arr)) if arr.size else 0.0,
            "batch_size": int(arr.size),
        }