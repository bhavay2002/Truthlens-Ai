from __future__ import annotations

import logging
from typing import Callable, Dict, List

import numpy as np

from src.explainability.utils_validation import validate_tokens_scores

logger = logging.getLogger(__name__)
PredictionFn = Callable[[str], Dict[str, float]]


class ExplanationMetrics:
    def __init__(self) -> None:
        logger.info("ExplanationMetrics initialized")

    @staticmethod
    def _extract_fake_probability(result: Dict[str, float]) -> float:
        if "fake_probability" not in result:
            raise KeyError("Prediction output must contain 'fake_probability'")
        return float(result["fake_probability"])

    @staticmethod
    def _sort_indices(scores: List[float]) -> List[int]:
        return list(np.argsort(np.asarray(scores))[::-1])

    @staticmethod
    def _validate(tokens: List[str], scores: List[float]) -> None:
        validate_tokens_scores(tokens, scores)

    def faithfulness(self, tokens, scores, predict_fn):
        self._validate(tokens, scores)
        base = self._extract_fake_probability(predict_fn(" ".join(tokens)))
        deltas = []
        for i in range(len(tokens)):
            perturbed = " ".join([t for j, t in enumerate(tokens) if j != i])
            val = self._extract_fake_probability(predict_fn(perturbed))
            deltas.append(base - val)
        if len(deltas) < 2:
            return 0.0
        corr = np.corrcoef(scores, deltas)[0, 1]
        return 0.0 if np.isnan(corr) else float(corr)

    def comprehensiveness(self, tokens, scores, predict_fn, top_k=5):
        self._validate(tokens, scores)
        base = self._extract_fake_probability(predict_fn(" ".join(tokens)))
        ranked = self._sort_indices(scores)[:top_k]
        perturbed = " ".join([t for i, t in enumerate(tokens) if i not in set(ranked)])
        new = self._extract_fake_probability(predict_fn(perturbed))
        return float(base - new)

    def sufficiency(self, tokens, scores, predict_fn, top_k=5):
        self._validate(tokens, scores)
        base = self._extract_fake_probability(predict_fn(" ".join(tokens)))
        ranked = self._sort_indices(scores)[:top_k]
        kept = [tokens[i] for i in sorted(ranked)]
        new = self._extract_fake_probability(predict_fn(" ".join(kept)))
        return float(base - new)

    def deletion_score(self, tokens, scores, predict_fn):
        self._validate(tokens, scores)
        ranked = self._sort_indices(scores)
        base = self._extract_fake_probability(predict_fn(" ".join(tokens)))
        current = tokens.copy()
        preds = []
        for idx in ranked:
            current[idx] = ""
            text = " ".join([t for t in current if t])
            preds.append(self._extract_fake_probability(predict_fn(text)))
        return float(base - np.mean(np.asarray(preds)))

    def insertion_score(self, tokens, scores, predict_fn):
        self._validate(tokens, scores)
        ranked = self._sort_indices(scores)
        slots = [""] * len(tokens)  # preserve original positions
        preds = []
        for idx in ranked:
            slots[idx] = tokens[idx]
            text = " ".join([t for t in slots if t])
            preds.append(self._extract_fake_probability(predict_fn(text)))
        return float(np.trapz(preds))

    def evaluate(self, tokens, scores, predict_fn):
        return {
            "faithfulness": self.faithfulness(tokens, scores, predict_fn),
            "comprehensiveness": self.comprehensiveness(tokens, scores, predict_fn),
            "sufficiency": self.sufficiency(tokens, scores, predict_fn),
            "deletion_score": self.deletion_score(tokens, scores, predict_fn),
            "insertion_score": self.insertion_score(tokens, scores, predict_fn),
        }
