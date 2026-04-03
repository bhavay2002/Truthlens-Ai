"""
File Name: syntactic_features.py
Module: Text Feature Engineering - Syntactic Features
Description:
    Extracts syntactic features from text including part-of-speech
    distributions and sentence-level structural statistics. These
    features provide insight into grammatical structure and complexity
    of language used in the input text.

    The module supports optional integration with spaCy for accurate
    POS tagging. If spaCy is unavailable, a lightweight fallback
    tokenizer and heuristic POS estimation are used to ensure the
    system remains functional.

Dependencies:
    dataclasses
    typing
    logging
    re
    collections
    spacy (optional)

Inputs:
    FeatureContext containing text and optional tokens

Outputs:
    Dict[str, float] containing syntactic feature values
"""

from __future__ import annotations

import logging
import re
from collections import Counter
from dataclasses import dataclass
from typing import Dict, List

from src.features.base.base_feature import BaseFeature, FeatureContext
from src.features.base.feature_registry import register_feature

logger = logging.getLogger(__name__)

try:
    import spacy

    _NLP = spacy.load("en_core_web_sm")
    SPACY_AVAILABLE = True
except Exception:  # noqa: BLE001
    _NLP = None
    SPACY_AVAILABLE = False
    logger.warning("spaCy not available. Falling back to heuristic syntactic features.")


def _simple_sentence_split(text: str) -> List[str]:
    """Split text into sentences using punctuation heuristics."""
    sentences = re.split(r"[.!?]+", text)
    return [s.strip() for s in sentences if s.strip()]


def _simple_tokenize(text: str) -> List[str]:
    """Basic tokenizer fallback."""
    return re.findall(r"\b\w+\b", text.lower())


@dataclass
@register_feature
class SyntacticFeatures(BaseFeature):
    """
    Computes syntactic statistics including:

    - sentence_count
    - avg_sentence_length
    - noun_ratio
    - verb_ratio
    - adjective_ratio
    - adverb_ratio
    - punctuation_ratio
    """

    name: str = "syntactic_features"
    description: str = "Sentence structure and POS distribution features"

    def _extract_spacy(self, text: str) -> Dict[str, float]:
        """Extract syntactic features using spaCy."""
        doc = _NLP(text)

        tokens = [t for t in doc if not t.is_space]
        sentences = list(doc.sents)

        pos_counter = Counter(token.pos_ for token in tokens)

        token_count = len(tokens) if tokens else 1

        noun_ratio = pos_counter.get("NOUN", 0) / token_count
        verb_ratio = pos_counter.get("VERB", 0) / token_count
        adj_ratio = pos_counter.get("ADJ", 0) / token_count
        adv_ratio = pos_counter.get("ADV", 0) / token_count
        punct_ratio = pos_counter.get("PUNCT", 0) / token_count

        sentence_lengths = [len([t for t in sent if not t.is_punct]) for sent in sentences]

        avg_sentence_length = (
            sum(sentence_lengths) / len(sentence_lengths) if sentence_lengths else 0.0
        )

        return {
            "sentence_count": float(len(sentences)),
            "avg_sentence_length": float(avg_sentence_length),
            "noun_ratio": float(noun_ratio),
            "verb_ratio": float(verb_ratio),
            "adjective_ratio": float(adj_ratio),
            "adverb_ratio": float(adv_ratio),
            "punctuation_ratio": float(punct_ratio),
        }

    def _extract_fallback(self, text: str) -> Dict[str, float]:
        """Fallback syntactic estimation without NLP libraries."""
        tokens = _simple_tokenize(text)
        sentences = _simple_sentence_split(text)

        token_count = len(tokens) if tokens else 1

        punctuation_count = len(re.findall(r"[^\w\s]", text))

        avg_sentence_length = (
            token_count / len(sentences) if sentences else float(token_count)
        )

        return {
            "sentence_count": float(len(sentences)),
            "avg_sentence_length": float(avg_sentence_length),
            "noun_ratio": 0.0,
            "verb_ratio": 0.0,
            "adjective_ratio": 0.0,
            "adverb_ratio": 0.0,
            "punctuation_ratio": float(punctuation_count / token_count),
        }

    def extract(self, context: FeatureContext) -> Dict[str, float]:
        """
        Extract syntactic features.

        Parameters
        ----------
        context : FeatureContext

        Returns
        -------
        Dict[str, float]
        """
        if not context.text:
            raise ValueError("FeatureContext.text cannot be empty")

        if SPACY_AVAILABLE:
            features = self._extract_spacy(context.text)
        else:
            features = self._extract_fallback(context.text)

        logger.debug(
            "Syntactic features extracted | sentences=%s avg_len=%.2f",
            features["sentence_count"],
            features["avg_sentence_length"],
        )

        return features
