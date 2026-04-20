from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Dict, List, Set, Any  # ✅ FIXED

import numpy as np
from spacy.language import Language
from spacy.tokens import Doc

from src.analysis._nlp import get_nlp
from src.analysis._text_features import extract_alpha_lemmas, build_counter
from src.analysis.feature_schema import INFORMATION_DENSITY_KEYS, make_vector

logger = logging.getLogger(__name__)


# =========================================================
# CONFIG
# =========================================================

@dataclass(slots=True)
class InformationDensityConfig:
    spacy_model: str = "en_core_web_sm"
    disable_components: tuple = ()


# =========================================================
# ANALYZER
# =========================================================

class InformationDensityAnalyzer:

    FACTUAL_TERMS: Set[str] = {
        "according", "reported", "confirmed", "stated", "announced", "revealed",
        "showed", "demonstrated", "proved", "established", "documented", "verified",
        "found", "concluded", "published", "researched", "studied", "measured",
        "recorded", "observed", "evidence", "data", "statistics", "research",
        "study", "report", "analysis", "survey", "census", "experiment",
    }

    OPINION_TERMS: Set[str] = {
        "believe", "think", "feel", "argue", "contend", "suggest", "claim",
        "assert", "maintain", "insist", "seem", "appear", "presumably", "probably",
        "likely", "apparently", "supposedly", "allegedly", "arguably",
        "in my view", "in my opinion", "it seems", "i believe", "we think",
    }

    CLAIM_TERMS: Set[str] = {
        "claims", "alleges", "asserts", "declares", "states", "argues",
        "contends", "maintains", "insists", "purports", "denies", "admits",
        "acknowledges", "concedes", "charges", "accuses", "blames",
        "according to", "sources say", "reportedly", "allegedly",
    }

    RHETORICAL_TERMS: Set[str] = {
        "obviously", "clearly", "undeniably", "unquestionably", "certainly",
        "absolutely", "definitely", "surely", "indeed", "of course",
        "needless to say", "it is clear", "everyone knows", "nobody",
        "always", "never", "every", "all", "none", "impossible", "inevitable",
    }

    EMOTIONAL_TERMS: Set[str] = {
        "outrageous", "shocking", "disgusting", "horrifying", "terrible",
        "devastating", "catastrophic", "alarming", "dangerous", "frightening",
        "wonderful", "amazing", "incredible", "fantastic", "brilliant",
        "heartbreaking", "tragic", "disastrous", "explosive", "crisis",
        "threat", "attack", "destroy", "collapse", "panic", "fear", "rage",
    }

    MODAL_TERMS: Set[str] = {
        "should", "would", "could", "might", "must", "may", "shall", "will",
        "ought", "need", "dare", "used to", "had better", "would rather",
    }

    RHETORICAL_PATTERN = re.compile(r"[!?]+", re.UNICODE)

    # ----------------------------------------------------

    def __init__(self, config: InformationDensityConfig | None = None):
        self.config = config or InformationDensityConfig()

        self.nlp: Language = get_nlp(
            self.config.spacy_model,
            disable=self.config.disable_components,
        )

        self._compiled_patterns = self._compile_lexicons()

        logger.info("InformationDensityAnalyzer initialized")

    # =========================================================
    # MAIN
    # =========================================================

    def analyze(self, text: str) -> Dict[str, float]:
        if not isinstance(text, str):
            raise ValueError("Input text must be a string")

        text = text.strip()
        if not text:
            raise ValueError("Input text must be non-empty")

        return self.analyze_doc(self.nlp(text))

    def analyze_doc(self, doc: Doc) -> Dict[str, float]:

        tokens: List[str] = extract_alpha_lemmas(doc)
        token_counts = build_counter(tokens)
        n_tokens = max(len(tokens), 1)

        text_lower = doc.text.lower()

        features: Dict[str, float] = {}

        for key in ["factual", "opinion", "claim", "rhetorical", "emotion", "modal"]:
            features[f"{key}_density"] = self._compute_density(
                key, token_counts, text_lower, n_tokens
            )

        features.update(self._punctuation_rhetoric(text_lower))
        features.update(self._information_emotion_ratio(features))

        return features

    # =========================================================
    # CORE
    # =========================================================

    def _compile_lexicons(self) -> Dict[str, Dict[str, Any]]:
        def split_terms(terms: Set[str]):
            single = {t for t in terms if " " not in t}
            phrases = {t for t in terms if " " in t}
            patterns = [re.compile(rf"\b{re.escape(p)}\b") for p in phrases]
            return {"single": single, "patterns": patterns}

        return {
            "factual": split_terms(self.FACTUAL_TERMS),
            "opinion": split_terms(self.OPINION_TERMS),
            "claim": split_terms(self.CLAIM_TERMS),
            "rhetorical": split_terms(self.RHETORICAL_TERMS),
            "emotion": split_terms(self.EMOTIONAL_TERMS),
            "modal": split_terms(self.MODAL_TERMS),
        }

    def _compute_density(
        self,
        key: str,
        token_counts: Dict[str, int],
        text_lower: str,
        n_tokens: int,
    ) -> float:

        cfg = self._compiled_patterns[key]

        hits = sum(token_counts.get(tok, 0) for tok in cfg["single"])

        for pattern in cfg["patterns"]:
            hits += len(pattern.findall(text_lower))

        density = hits / n_tokens

        # ✅ clamp to valid range
        density = float(np.clip(density, 0.0, 1.0))

        return density

    # =========================================================
    # AUX
    # =========================================================

    def _punctuation_rhetoric(self, text: str) -> Dict[str, float]:
        matches = self.RHETORICAL_PATTERN.findall(text)
        length = max(len(text.split()), 1)

        density = len(matches) / length
        density = float(np.clip(density, 0.0, 1.0))

        return {"rhetorical_punctuation_density": density}

    def _information_emotion_ratio(
        self,
        features: Dict[str, float]
    ) -> Dict[str, float]:

        factual = features.get("factual_density", 0.0)
        emotion = features.get("emotion_density", 0.0)

        eps = 1e-9

        raw_ratio = factual / max(emotion, eps)
        raw_ratio = float(np.clip(raw_ratio, 0.0, 10.0))

        normalized_ratio = raw_ratio / 10.0

        return {
            "information_emotion_ratio": raw_ratio,
            "information_emotion_ratio_normalized": normalized_ratio
        }


# =========================================================
# VECTOR
# =========================================================

def information_density_vector(features: Dict[str, float]) -> np.ndarray:
    return make_vector(features, INFORMATION_DENSITY_KEYS)