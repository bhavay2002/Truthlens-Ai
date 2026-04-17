"""
File Name: syntactic_features.py
Module: Text Feature Engineering - Syntactic Features
"""

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


def _simple_sentence_split(text: str) -> List[str]:
    sentences = re.split(r"[.!?]+", text)
    return [s.strip() for s in sentences if s.strip()]


def _simple_tokenize(text: str) -> List[str]:
    return re.findall(r"\b\w+\b", text.lower())


@dataclass
@register_feature
class SyntacticFeatures(BaseFeature):
    name: str = "syntactic_features"
    description: str = "Sentence structure and POS distribution features"

    _nlp: Any = field(default=None, init=False, repr=False)
    _spacy_available: bool = field(default=False, init=False, repr=False)

    def initialize(self) -> None:
        if self._nlp is not None or self._spacy_available:
            return
        try:
            import spacy
            self._nlp = spacy.load("en_core_web_sm")
            self._spacy_available = True
        except Exception:  # noqa: BLE001
            self._nlp = None
            self._spacy_available = False
            logger.warning("spaCy not available. Falling back to heuristic syntactic features.")

    def _extract_spacy_doc(self, doc) -> Dict[str, float]:
        tokens = [t for t in doc if not t.is_space]
        sentences = list(doc.sents)

        pos_counter = Counter(token.pos_ for token in tokens)
        token_count = len(tokens) if tokens else 1

        noun_ratio = pos_counter.get("NOUN", 0) / token_count
        verb_ratio = pos_counter.get("VERB", 0) / token_count
        adj_ratio = pos_counter.get("ADJ", 0) / token_count
        adv_ratio = pos_counter.get("ADV", 0) / token_count
        punct_ratio = pos_counter.get("PUNCT", 0) / token_count

        sentence_lengths = np.array(
            [len([t for t in sent if not t.is_punct]) for sent in sentences],
            dtype=np.float32,
        )
        avg_sentence_length = float(sentence_lengths.mean()) if sentence_lengths.size else 0.0

        return {
            "sentence_count": float(len(sentences)),
            "avg_sentence_length": float(avg_sentence_length),
            "noun_ratio": float(noun_ratio),
            "verb_ratio": float(verb_ratio),
            "adjective_ratio": float(adj_ratio),
            "adverb_ratio": float(adv_ratio),
            "punctuation_ratio": float(punct_ratio),
        }

    def _extract_spacy(self, text: str) -> Dict[str, float]:
        doc = self._nlp(text)
        return self._extract_spacy_doc(doc)

    def _extract_fallback(self, text: str) -> Dict[str, float]:
        tokens = _simple_tokenize(text)
        sentences = _simple_sentence_split(text)

        token_count = len(tokens) if tokens else 1
        punctuation_count = len(re.findall(r"[^\w\s]", text))

        avg_sentence_length = token_count / len(sentences) if sentences else float(token_count)

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
        if not isinstance(context.text, str):
            raise TypeError("FeatureContext.text must be a string")
        if not context.text.strip():
            return {}

        self.initialize()
        if self._spacy_available and self._nlp is not None:
            features = self._extract_spacy(context.text)
        else:
            features = self._extract_fallback(context.text)

        logger.debug(
            "Syntactic features extracted | sentences=%s avg_len=%.2f",
            features["sentence_count"],
            features["avg_sentence_length"],
        )
        return features

    def extract_batch(self, contexts: List[FeatureContext]) -> List[Dict[str, float]]:
        if not contexts:
            return []

        self.initialize()
        if self._spacy_available and self._nlp is not None:
            texts = [c.text if isinstance(c.text, str) else "" for c in contexts]
            docs = list(self._nlp.pipe(texts, batch_size=32))
            return [self._extract_spacy_doc(doc) for doc in docs]

        return [self.extract(context) for context in contexts]