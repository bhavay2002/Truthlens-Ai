# src/features/propaganda_features.py

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
# Lexicons (reuse yours)
# ---------------------------------------------------------

NAME_CALLING = {...}
FEAR_APPEAL = {...}
EXAGGERATION = {...}
GLITTERING_GENERALITIES = {...}
US_VS_THEM = {...}
AUTHORITY_APPEAL = {...}
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
class PropagandaFeatures(BaseFeature):

    name: str = "propaganda_features"
    group: str = "propaganda"
    description: str = "Normalized propaganda feature signals"

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
            "name_calling": _ratio(counter, NAME_CALLING, n),
            "fear": _ratio(counter, FEAR_APPEAL, n),
            "exaggeration": _ratio(counter, EXAGGERATION, n),
            "glitter": _ratio(counter, GLITTERING_GENERALITIES, n),
            "us_vs_them": _ratio(counter, US_VS_THEM, n),
            "authority": _ratio(counter, AUTHORITY_APPEAL, n),
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
        # INTENSITY
        # -------------------------

        intensity = float(np.linalg.norm(values))

        # -------------------------
        # ENTROPY
        # -------------------------

        if probs.sum() > 0:
            entropy_raw = -np.sum(probs * np.log(probs + EPS))
            entropy = entropy_raw / (np.log(len(probs)) + EPS)
        else:
            entropy = 0.0

        # -------------------------
        # DIVERSITY
        # -------------------------

        diversity = float(np.count_nonzero(values) / len(values))

        # -------------------------
        # RHETORIC
        # -------------------------

        rhetoric = (text.count("!") + text.count("?")) / (n + EPS)

        # -------------------------
        # CAPS EMPHASIS
        # -------------------------

        caps_tokens = sum(
            1 for w in text.split() if w.isupper() and len(w) > 2
        )
        caps_ratio = caps_tokens / (n + EPS)

        # -------------------------
        # OUTPUT
        # -------------------------

        return {
            "propaganda_name_calling": self._safe(dist["name_calling"]),
            "propaganda_fear": self._safe(dist["fear"]),
            "propaganda_exaggeration": self._safe(dist["exaggeration"]),
            "propaganda_glitter": self._safe(dist["glitter"]),
            "propaganda_us_vs_them": self._safe(dist["us_vs_them"]),
            "propaganda_authority": self._safe(dist["authority"]),
            "propaganda_intensifier": self._safe(dist["intensifier"]),

            "propaganda_intensity": self._safe(intensity),
            "propaganda_entropy": self._safe(entropy),
            "propaganda_diversity": self._safe(diversity),

            "propaganda_rhetoric": self._safe(rhetoric),
            "propaganda_caps_ratio": self._safe(caps_ratio),
        }

    # -----------------------------------------------------

    def _safe(self, v: float) -> float:
        if not np.isfinite(v):
            return 0.0
        return float(np.clip(v, 0.0, MAX_CLIP))