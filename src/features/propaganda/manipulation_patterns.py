# src/features/manipulation_patterns.py

from __future__ import annotations

import logging
import re
from collections import Counter
from dataclasses import dataclass
from typing import Dict, List, Set

import numpy as np

from src.features.base.base_feature import BaseFeature, FeatureContext
from src.features.base.feature_registry import register_feature

logger = logging.getLogger(__name__)

EPS = 1e-8
MAX_CLIP = 1.0


# ---------------------------------------------------------
# Tokenization
# ---------------------------------------------------------

TOKEN_PATTERN = re.compile(r"[A-Za-z']+")

def _tokenize(text: str) -> List[str]:
    return TOKEN_PATTERN.findall(text.lower())


# ---------------------------------------------------------
# Lexicons (same as yours)
# ---------------------------------------------------------

URGENCY_TERMS = {...}
FEAR_TERMS = {...}
BLAME_TERMS = {...}
SCAPEGOAT_TERMS = {...}
ABSOLUTE_TERMS = {...}
CONSPIRACY_TERMS = {...}
FALSE_DILEMMA_TERMS = {...}
EXAGGERATION_TERMS = {...}
INTENSIFIERS = {...}


# ---------------------------------------------------------
# Helper
# ---------------------------------------------------------

def _ratio(counter: Counter, lexicon: Set[str], total: int) -> float:
    return sum(counter.get(w, 0) for w in lexicon) / (total + EPS)


# ---------------------------------------------------------
# Feature
# ---------------------------------------------------------

@dataclass
@register_feature
class ManipulationPatterns(BaseFeature):

    name: str = "manipulation_patterns"
    group: str = "propaganda"
    description: str = "Normalized manipulation pattern features"

    # -----------------------------------------------------

    def extract(self, context: FeatureContext) -> Dict[str, float]:

        text = context.text.strip()
        if not text:
            return {}

        tokens = context.tokens or _tokenize(text)
        n = len(tokens)

        if n == 0:
            return {}

        counter = Counter(tokens)

        raw = {
            "urgency": _ratio(counter, URGENCY_TERMS, n),
            "fear": _ratio(counter, FEAR_TERMS, n),
            "blame": _ratio(counter, BLAME_TERMS, n),
            "scapegoat": _ratio(counter, SCAPEGOAT_TERMS, n),
            "absolute": _ratio(counter, ABSOLUTE_TERMS, n),
            "conspiracy": _ratio(counter, CONSPIRACY_TERMS, n),
            "dilemma": _ratio(counter, FALSE_DILEMMA_TERMS, n),
            "exaggeration": _ratio(counter, EXAGGERATION_TERMS, n),
            "intensifier": _ratio(counter, INTENSIFIERS, n),
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
        # INTENSITY (STRONGER)
        # -------------------------

        intensity = float(np.linalg.norm(values))

        # -------------------------
        # ENTROPY (CRITICAL)
        # -------------------------

        if probs.sum() > 0:
            entropy_raw = -np.sum(probs * np.log(probs + EPS))
            entropy = entropy_raw / (np.log(len(probs)) + EPS)
        else:
            entropy = 0.0

        # -------------------------
        # DIVERSITY (WEIGHTED)
        # -------------------------

        diversity = float(np.count_nonzero(values) / len(values))

        # -------------------------
        # RHETORIC (FIXED)
        # -------------------------

        exclam = text.count("!")
        questions = text.count("?")
        rhetoric = (exclam + questions) / (n + EPS)

        # -------------------------
        # CAPS EMPHASIS (FIXED)
        # -------------------------

        caps_tokens = sum(
            1 for w in text.split() if w.isupper() and len(w) > 2
        )
        caps_ratio = caps_tokens / (n + EPS)

        # -------------------------
        # OUTPUT
        # -------------------------

        return {
            "manipulation_urgency": self._safe(dist["urgency"]),
            "manipulation_fear": self._safe(dist["fear"]),
            "manipulation_blame": self._safe(dist["blame"]),
            "manipulation_scapegoat": self._safe(dist["scapegoat"]),
            "manipulation_absolute": self._safe(dist["absolute"]),
            "manipulation_conspiracy": self._safe(dist["conspiracy"]),
            "manipulation_false_dilemma": self._safe(dist["dilemma"]),
            "manipulation_exaggeration": self._safe(dist["exaggeration"]),
            "manipulation_intensifier": self._safe(dist["intensifier"]),

            "manipulation_intensity": self._safe(intensity),
            "manipulation_entropy": self._safe(entropy),
            "manipulation_diversity": self._safe(diversity),

            "manipulation_rhetoric": self._safe(rhetoric),
            "manipulation_caps_emphasis": self._safe(caps_ratio),
        }

    # -----------------------------------------------------

    def _safe(self, v: float) -> float:
        if not np.isfinite(v):
            return 0.0
        return float(np.clip(v, 0.0, MAX_CLIP))