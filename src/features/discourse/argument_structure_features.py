# src/features/argument_structure_features.py

from __future__ import annotations

import logging
from collections import Counter
from dataclasses import dataclass
from typing import Dict, Set

import numpy as np

from src.features.base.base_feature import BaseFeature, FeatureContext
from src.features.base.feature_registry import register_feature
from src.features.base.numerics import normalized_entropy
from src.features.base.tokenization import ensure_tokens_word

logger = logging.getLogger(__name__)

EPS = 1e-8
MAX_CLIP = 1.0


# ---------------------------------------------------------
# Lexicons
# ---------------------------------------------------------

CLAIM_MARKERS = {"therefore","thus","clearly","obviously","conclude","shows"}
PREMISE_MARKERS = {"because","since","given","as","assuming"}
EVIDENCE_MARKERS = {"evidence","study","data","report","research","analysis"}
COUNTERARGUMENT_MARKERS = {"however","although","but","nevertheless","yet"}

INTERROGATIVES = {"why","how","what","who"}


# ---------------------------------------------------------
# Feature
# ---------------------------------------------------------

@dataclass
@register_feature
class ArgumentStructureFeatures(BaseFeature):

    name: str = "argument_structure_features"
    group: str = "argument"
    description: str = "Normalized argument structure features"

    # -----------------------------------------------------

    def extract(self, context: FeatureContext) -> Dict[str, float]:

        text = context.text.strip()
        if not text:
            return {}

        tokens = ensure_tokens_word(context, text)
        n = len(tokens)

        if n == 0:
            return {}

        counter = Counter(tokens)

        def ratio(lexicon: Set[str]) -> float:
            return sum(counter.get(w, 0) for w in lexicon) / (n + EPS)

        raw = {
            "claim": ratio(CLAIM_MARKERS),
            "premise": ratio(PREMISE_MARKERS),
            "evidence": ratio(EVIDENCE_MARKERS),
            "counter": ratio(COUNTERARGUMENT_MARKERS),
        }

        # -------------------------
        # NORMALIZED DISTRIBUTION
        # -------------------------

        values = np.array(list(raw.values()), dtype=np.float32)
        total = values.sum()

        if total < EPS:
            dist = {k: 0.0 for k in raw}
        else:
            norm = values / (total + EPS)
            dist = dict(zip(raw.keys(), norm.astype(float)))

        probs = np.array(list(dist.values()), dtype=np.float32)

        # -------------------------
        # INTENSITY (ARGUMENT STRENGTH)
        # -------------------------

        intensity = float(np.linalg.norm(values))

        # -------------------------
        # ENTROPY (CRITICAL)
        # -------------------------

        entropy = normalized_entropy(probs)

        # -------------------------
        # RHETORICAL QUESTIONS (FIXED)
        # -------------------------

        question_marks = text.count("?")
        interrogative_hits = sum(counter.get(w, 0) for w in INTERROGATIVES)

        rhetorical = (question_marks + interrogative_hits) / (n + EPS)

        # -------------------------
        # ARGUMENT BALANCE
        # -------------------------

        balance = 1.0 - float(np.std(probs))

        # -------------------------
        # OUTPUT
        # -------------------------

        return {
            "arg_claim": self._safe(dist["claim"]),
            "arg_premise": self._safe(dist["premise"]),
            "arg_evidence": self._safe(dist["evidence"]),
            "arg_counter": self._safe(dist["counter"]),

            "arg_intensity": self._safe(intensity),
            "arg_entropy": self._safe(entropy),
            "arg_balance": self._safe(balance),

            "arg_rhetorical": self._safe(rhetorical),
        }

    # -----------------------------------------------------

    def _safe(self, v: float) -> float:
        if not np.isfinite(v):
            return 0.0
        return float(np.clip(v, 0.0, MAX_CLIP))