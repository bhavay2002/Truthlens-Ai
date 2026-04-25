# src/features/syntactic_features.py

from __future__ import annotations

import logging
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Dict, List, Any

import numpy as np

from src.features.base.base_feature import BaseFeature, FeatureContext
from src.features.base.feature_registry import register_feature

logger = logging.getLogger(__name__)

EPS = 1e-8
MAX_CLIP = 1.0


# ---------------------------------------------------------
# Helpers
# ---------------------------------------------------------

def _simple_sentence_split(text: str) -> List[str]:
    return [s.strip() for s in re.split(r"[.!?]+", text) if s.strip()]


def _simple_tokenize(text: str) -> List[str]:
    return re.findall(r"\b\w+\b", text.lower())


# ---------------------------------------------------------
# Feature
# ---------------------------------------------------------

@dataclass
@register_feature
class SyntacticFeatures(BaseFeature):

    name: str = "syntactic_features"
    group: str = "syntactic"
    description: str = "Advanced syntactic structure features"

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
            logger.warning("spaCy unavailable → fallback mode")

    # -----------------------------------------------------
    # spaCy version
    # -----------------------------------------------------

    def _extract_spacy_doc(self, doc) -> Dict[str, float]:

        tokens = [t for t in doc if not t.is_space]
        n = len(tokens) or 1

        # -------------------------
        # POS DISTRIBUTION
        # -------------------------

        pos_counts = Counter(t.pos_ for t in tokens)

        pos_keys = ["NOUN", "VERB", "ADJ", "ADV"]
        pos_vals = np.array([pos_counts.get(k, 0) for k in pos_keys], dtype=np.float32)

        pos_probs = pos_vals / (pos_vals.sum() + EPS)

        # entropy
        entropy_raw = -np.sum(pos_probs * np.log(pos_probs + EPS))
        pos_entropy = entropy_raw / np.log(len(pos_probs) + EPS)

        # -------------------------
        # SENTENCE STRUCTURE
        # -------------------------

        sentences = list(doc.sents)

        lengths = np.array(
            [len([t for t in s if not t.is_punct]) for s in sentences],
            dtype=np.float32,
        )

        avg_len = float(lengths.mean()) if lengths.size else 0.0
        std_len = float(lengths.std()) if lengths.size else 0.0

        # normalized dispersion
        dispersion = std_len / (avg_len + EPS)

        # entropy of sentence lengths
        if lengths.size > 1:
            probs = lengths / (lengths.sum() + EPS)
            ent_raw = -np.sum(probs * np.log(probs + EPS))
            sent_entropy = ent_raw / np.log(len(probs) + EPS)
        else:
            sent_entropy = 0.0

        # -------------------------
        # SYNTACTIC COMPLEXITY
        # -------------------------

        depths = []

        for token in tokens:
            depth = 0
            head = token
            while head.head != head:
                depth += 1
                head = head.head
                if depth > 20:
                    break
            depths.append(depth)

        complexity = float(np.mean(depths)) if depths else 0.0

        # -------------------------
        # COORDINATION / SUBORDINATION
        # -------------------------

        conj = sum(1 for t in tokens if t.dep_ == "conj")
        subord = sum(1 for t in tokens if t.dep_ in {"ccomp", "advcl", "relcl"})

        coord_ratio = conj / (n + EPS)
        subord_ratio = subord / (n + EPS)

        # -------------------------
        # OUTPUT
        # -------------------------

        return {
            "syn_pos_entropy": self._safe(pos_entropy),

            "syn_sentence_avg_len": self._safe(avg_len / 50.0),
            "syn_sentence_dispersion": self._safe(dispersion),
            "syn_sentence_entropy": self._safe(sent_entropy),

            "syn_complexity": self._safe(complexity / 10.0),

            "syn_coordination": self._safe(coord_ratio),
            "syn_subordination": self._safe(subord_ratio),
        }

    # -----------------------------------------------------
    # fallback
    # -----------------------------------------------------

    def _extract_fallback(self, text: str) -> Dict[str, float]:

        tokens = _simple_tokenize(text)
        sentences = _simple_sentence_split(text)

        n = len(tokens) or 1

        avg_len = n / len(sentences) if sentences else n

        return {
            "syn_pos_entropy": 0.0,
            "syn_sentence_avg_len": self._safe(avg_len / 50.0),
            "syn_sentence_dispersion": 0.0,
            "syn_sentence_entropy": 0.0,
            "syn_complexity": 0.0,
            "syn_coordination": 0.0,
            "syn_subordination": 0.0,
        }

    # -----------------------------------------------------

    def extract(self, context: FeatureContext) -> Dict[str, float]:

        text = context.text.strip()
        if not text:
            return {}

        self.initialize()

        if self._spacy_available and self._nlp is not None:
            doc = self._nlp(text)
            return self._extract_spacy_doc(doc)

        return self._extract_fallback(text)

    # -----------------------------------------------------

    def _safe(self, v: float) -> float:
        if not np.isfinite(v):
            return 0.0
        return float(np.clip(v, 0.0, MAX_CLIP))