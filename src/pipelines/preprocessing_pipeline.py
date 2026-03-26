"""
File Name: preprocessing_pipeline.py
Module: Data Processing - Text Preprocessing Pipeline
Description:
    Implements the text preprocessing pipeline used in the TruthLens AI system.
    The module performs normalization, cleaning, token preparation, sentence
    segmentation, and optional stopword removal. It provides standardized text
    outputs used by downstream NLP modules such as bias detection, emotion
    analysis, narrative analysis, and transformer-based models.

Dependencies:
    logging
    typing
    re
    spacy

Inputs:
    Raw text string

Outputs:
    Preprocessed text structure containing normalized text, tokens, and sentences
"""

import logging
import re
from typing import Dict, List

import spacy


logger = logging.getLogger(__name__)


class PreprocessingPipeline:
    """
    Standardized preprocessing pipeline for textual inputs.
    """

    def __init__(self, spacy_model: str = "en_core_web_sm") -> None:
        """Initialize NLP pipeline used for preprocessing."""

        try:
            self.nlp = spacy.load(spacy_model)
        except Exception as exc:
            logger.exception("spaCy model loading failed")
            raise RuntimeError("Failed to load spaCy model") from exc

        logger.info("PreprocessingPipeline initialized")

    def preprocess(self, text: str) -> Dict[str, List[str]]:
        """Run the preprocessing pipeline on input text."""

        if not isinstance(text, str) or not text.strip():
            raise ValueError("text must be a non-empty string")

        try:
            normalized_text = self._normalize_text(text)

            doc = self.nlp(normalized_text)

        except Exception as exc:
            logger.exception("Text preprocessing failed")
            raise RuntimeError("Preprocessing pipeline failed") from exc

        tokens = self._extract_tokens(doc)

        sentences = self._extract_sentences(doc)

        lemmas = self._extract_lemmas(doc)

        result = {
            "normalized_text": normalized_text,
            "tokens": tokens,
            "lemmas": lemmas,
            "sentences": sentences,
        }

        return result

    def _normalize_text(self, text: str) -> str:
        """Normalize text by removing excessive whitespace and special characters."""

        text = text.strip()

        text = re.sub(r"\s+", " ", text)

        text = re.sub(r"[^\w\s\.\,\!\?\-']", "", text)

        return text

    def _extract_tokens(self, doc) -> List[str]:
        """Extract token list from document."""

        tokens = [token.text.lower() for token in doc if not token.is_space]

        return tokens

    def _extract_lemmas(self, doc) -> List[str]:
        """Extract lemma list from document."""

        lemmas = [token.lemma_.lower() for token in doc if not token.is_space]

        return lemmas

    def _extract_sentences(self, doc) -> List[str]:
        """Extract sentence list from document."""

        sentences = [sent.text.strip() for sent in doc.sents if sent.text.strip()]

        return sentences