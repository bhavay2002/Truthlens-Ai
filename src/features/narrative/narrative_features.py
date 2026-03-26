"""
File Name: narrative_features.py
Module: Narrative Analysis - Feature Extraction
Description:
    Extracts narrative-level linguistic and structural features used by the
    TruthLens AI system. The module identifies narrative framing signals,
    actor-action structures, temporal progression, conflict indicators,
    and storytelling patterns that frequently appear in political narratives,
    propaganda, and ideological messaging.

Dependencies:
    logging
    re
    typing
    collections
    numpy
    spacy

Inputs:
    Raw text string

Outputs:
    Dictionary of narrative-level features and numerical feature vector
"""

import logging
from typing import Dict, List

import numpy as np
import spacy


logger = logging.getLogger(__name__)


class NarrativeFeatureExtractor:
    """
    Extracts narrative framing and storytelling features from text.
    """

    TEMPORAL_MARKERS = {
        "then",
        "later",
        "after",
        "before",
        "finally",
        "eventually",
        "suddenly",
        "meanwhile",
        "during",
        "when",
    }

    CONFLICT_MARKERS = {
        "fight",
        "battle",
        "attack",
        "oppose",
        "threat",
        "crisis",
        "conflict",
        "war",
        "struggle",
    }

    CAUSAL_MARKERS = {
        "because",
        "therefore",
        "thus",
        "hence",
        "since",
        "as",
    }

    def __init__(self, spacy_model: str = "en_core_web_sm") -> None:
        """Initialize narrative feature extractor."""

        try:
            self.nlp = spacy.load(spacy_model)
        except Exception as exc:
            logger.exception("spaCy model loading failed")
            raise RuntimeError("Failed to load spaCy model") from exc

        logger.info("NarrativeFeatureExtractor initialized")

    def extract(self, text: str) -> Dict[str, float]:
        """Extract narrative-related linguistic features."""

        if not isinstance(text, str) or not text.strip():
            raise ValueError("Input text must be a non-empty string")

        try:
            doc = self.nlp(text)
        except Exception as exc:
            logger.exception("spaCy text processing failed")
            raise RuntimeError("Text processing failed") from exc

        tokens = [token.text.lower() for token in doc if token.is_alpha]

        features: Dict[str, float] = {}

        features.update(self._actor_features(doc))
        features.update(self._temporal_features(tokens))
        features.update(self._conflict_features(tokens))
        features.update(self._causal_features(tokens))
        features.update(self._event_structure(doc))

        return features

    def _actor_features(self, doc) -> Dict[str, float]:
        """Detect narrative actors through named entities and subjects."""

        entities = [ent.text for ent in doc.ents if ent.label_ in {"PERSON", "ORG", "GPE"}]

        subjects = [token for token in doc if token.dep_ == "nsubj"]

        total_tokens = max(len(doc), 1)

        return {
            "narrative_entity_ratio": float(len(entities) / total_tokens),
            "narrative_subject_ratio": float(len(subjects) / total_tokens),
        }

    def _temporal_features(self, tokens: List[str]) -> Dict[str, float]:
        """Measure temporal progression indicators."""

        if not tokens:
            return {"narrative_temporal_ratio": 0.0}

        count = sum(1 for token in tokens if token in self.TEMPORAL_MARKERS)

        ratio = count / max(len(tokens), 1)

        return {"narrative_temporal_ratio": float(ratio)}

    def _conflict_features(self, tokens: List[str]) -> Dict[str, float]:
        """Detect narrative conflict signals."""

        if not tokens:
            return {"narrative_conflict_ratio": 0.0}

        count = sum(1 for token in tokens if token in self.CONFLICT_MARKERS)

        ratio = count / max(len(tokens), 1)

        return {"narrative_conflict_ratio": float(ratio)}

    def _causal_features(self, tokens: List[str]) -> Dict[str, float]:
        """Detect causal reasoning structures in narratives."""

        if not tokens:
            return {"narrative_causal_ratio": 0.0}

        count = sum(1 for token in tokens if token in self.CAUSAL_MARKERS)

        ratio = count / max(len(tokens), 1)

        return {"narrative_causal_ratio": float(ratio)}

    def _event_structure(self, doc) -> Dict[str, float]:
        """Estimate narrative event density using verbs."""

        verbs = [token for token in doc if token.pos_ == "VERB"]

        total_tokens = max(len(doc), 1)

        return {
            "narrative_event_density": float(len(verbs) / total_tokens)
        }


def narrative_feature_vector(features: Dict[str, float]) -> np.ndarray:
    """Convert narrative feature dictionary into numeric vector."""

    if not isinstance(features, dict) or not features:
        raise ValueError("features must be a non-empty dictionary")

    try:
        vector = np.array(list(features.values()), dtype=np.float32)
        return vector
    except Exception as exc:
        logger.exception("Narrative vector conversion failed")
        raise RuntimeError("Failed to convert narrative features") from exc
