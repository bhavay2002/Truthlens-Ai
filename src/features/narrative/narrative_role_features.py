# src/features/narrative_role_features.py

from __future__ import annotations

import logging
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Dict, List, Set, Any

import numpy as np

from src.features.base.base_feature import BaseFeature, FeatureContext
from src.features.base.feature_registry import register_feature

logger = logging.getLogger(__name__)

EPS = 1e-8
MAX_CLIP = 1.0


# ---------------------------------------------------------
# Tokenization
# ---------------------------------------------------------

def _tokenize(text: str) -> List[str]:
    return re.findall(r"\b\w+\b", text.lower())


# ---------------------------------------------------------
# Lexicons
# ---------------------------------------------------------

HERO_TERMS: Set[str] = {...}
VILLAIN_TERMS: Set[str] = {...}
VICTIM_TERMS: Set[str] = {...}
POLARIZATION_TERMS: Set[str] = {...}


# ---------------------------------------------------------
# Feature
# ---------------------------------------------------------

@dataclass
@register_feature
class NarrativeRoleFeatures(BaseFeature):

    name: str = "narrative_role_features"
    group: str = "narrative"
    description: str = "Normalized narrative role modeling"

    _nlp: Any = field(default=None, init=False, repr=False)
    _spacy_available: bool = field(default=False, init=False, repr=False)

    # -----------------------------------------------------

    def initialize(self) -> None:
        if self._nlp is not None or self._spacy_available:
            return
        try:
            import spacy
            self._nlp = spacy.load("en_core_web_sm")
            self._spacy_available = True
        except Exception:
            self._nlp = None
            self._spacy_available = False
            logger.warning("spaCy unavailable. Using fallback.")

    # -----------------------------------------------------

    def _entity_density(self, text: str) -> float:
        self.initialize()

        if not self._spacy_available or self._nlp is None:
            return 0.0

        doc = self._nlp(text)
        return len(doc.ents) / max(len(doc), 1)

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

        def ratio(lexicon: Set[str]) -> float:
            return sum(counter.get(w, 0) for w in lexicon) / (n + EPS)

        raw = {
            "hero": ratio(HERO_TERMS),
            "villain": ratio(VILLAIN_TERMS),
            "victim": ratio(VICTIM_TERMS),
        }

        polarization = ratio(POLARIZATION_TERMS)

        # -------------------------
        # ROLE DISTRIBUTION (CRITICAL)
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
        # BALANCE (FIXED)
        # -------------------------

        balance = 1.0 - float(np.std(probs))

        # -------------------------
        # DIVERSITY (WEIGHTED)
        # -------------------------

        diversity = float(np.count_nonzero(values) / len(values))

        # -------------------------
        # ENTITY SIGNAL
        # -------------------------

        entity_density = self._entity_density(text)

        # -------------------------
        # OUTPUT
        # -------------------------

        # Names MUST match src/features/feature_schema.py:NARRATIVE_FEATURES.
        # These are FEATURE names (model inputs) and are intentionally
        # distinct from the LABEL columns ("hero", "villain", "victim")
        # declared in data_contracts.CONTRACTS["narrative"].
        return {
            "narrative_role_hero_ratio": self._safe(dist["hero"]),
            "narrative_role_villain_ratio": self._safe(dist["villain"]),
            "narrative_role_victim_ratio": self._safe(dist["victim"]),

            "narrative_role_polarization_ratio": self._safe(polarization),

            "narrative_role_intensity": self._safe(intensity),
            "narrative_role_entropy": self._safe(entropy),

            "narrative_role_balance": self._safe(balance),
            "narrative_role_diversity": self._safe(diversity),

            "narrative_entity_density": self._safe(entity_density),
        }

    # -----------------------------------------------------

    def _safe(self, v: float) -> float:
        if not np.isfinite(v):
            return 0.0
        return float(np.clip(v, 0.0, MAX_CLIP))