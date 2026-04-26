"""
File Name: preprocessing_pipeline.py 
Module: Data Processing - Text Preprocessing Pipeline
Description:
    Implements an advanced preprocessing pipeline for textual inputs used in the
    TruthLens AI system. The pipeline performs text normalization, cleaning,
    sentence segmentation, tokenization, lemma extraction, and language detection.
    It also supports scalable batch preprocessing and parallel processing to
    efficiently handle large volumes of textual data.
    
Dependencies:
    logging
    re
    typing
    dataclasses
    concurrent.futures
    spacy
    langdetect

Inputs:
    Raw text string or list of text strings

Outputs:
    Structured preprocessing results including normalized text, tokens, lemmas,
    sentences, and detected language
"""

from __future__ import annotations

import logging
import re
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from typing import List, Optional

import spacy
try:
    from langdetect import detect, LangDetectException
except ImportError:  # pragma: no cover - optional dependency
    class LangDetectException(Exception):
        """Fallback exception when langdetect is unavailable."""

    def detect(_text: str) -> str:
        return "unknown"


logger = logging.getLogger(__name__)

_NLP = None


def _get_nlp():
    global _NLP
    if _NLP is None:
        _NLP = spacy.load("en_core_web_sm")
    return _NLP


@dataclass
class PreprocessingResult:
    """
    Dataclass representing structured preprocessing output.
    """

    normalized_text: str
    tokens: List[str]
    lemmas: List[str]
    sentences: List[str]
    language: str


class PreprocessingPipeline:
    """
    Production-grade preprocessing pipeline supporting scalable text processing.
    """

    def __init__(
        self,
        spacy_model: str = "en_core_web_sm",
        max_workers: Optional[int] = None,
    ) -> None:
        """
        Initialize NLP pipeline used for preprocessing.

        Parameters
        ----------
        spacy_model : str
            spaCy model name.
        max_workers : Optional[int]
            Maximum number of parallel workers for batch preprocessing.
        """

        try:
            self.nlp = spacy.load(spacy_model)
        except Exception as exc:
            logger.exception("spaCy model loading failed")
            raise RuntimeError("Failed to load spaCy model") from exc

        self.max_workers = max_workers

        logger.info(
            "PreprocessingPipeline initialized",
            extra={"spacy_model": spacy_model, "max_workers": max_workers},
        )

    def preprocess(self, text: str) -> PreprocessingResult:
        """
        Run preprocessing pipeline on a single text input.

        Parameters
        ----------
        text : str
            Raw input text.

        Returns
        -------
        PreprocessingResult
        """

        if not isinstance(text, str) or not text.strip():
            raise ValueError("text must be a non-empty string")

        try:
            language = self._detect_language(text)

            normalized_text = self._normalize_text(text)

            doc = self.nlp(normalized_text)

            tokens = self._extract_tokens(doc)

            lemmas = self._extract_lemmas(doc)

            sentences = self._extract_sentences(doc)

        except Exception as exc:
            logger.exception("Text preprocessing failed")
            raise RuntimeError("Preprocessing pipeline failed") from exc

        return PreprocessingResult(
            normalized_text=normalized_text,
            tokens=tokens,
            lemmas=lemmas,
            sentences=sentences,
            language=language,
        )

    def preprocess_batch(
        self,
        texts: List[str],
        parallel: bool = True,
    ) -> List[PreprocessingResult]:
        """
        Preprocess a batch of texts.

        Parameters
        ----------
        texts : List[str]
            List of raw text inputs.
        parallel : bool
            Whether to use parallel processing.

        Returns
        -------
        List[PreprocessingResult]
        """

        if not isinstance(texts, list):
            raise ValueError("texts must be a list of strings")

        if not texts:
            return []

        if parallel:
            return self._parallel_preprocess(texts)

        return [self.preprocess(text) for text in texts]

    def _parallel_preprocess(self, texts: List[str]) -> List[PreprocessingResult]:
        """
        Run preprocessing using parallel workers.

        Parameters
        ----------
        texts : List[str]

        Returns
        -------
        List[PreprocessingResult]
        """

        results: List[Optional[PreprocessingResult]] = [None] * len(texts)

        with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {
                executor.submit(PreprocessingPipeline._process_text_static, text): idx
                for idx, text in enumerate(texts)
            }

            for future in as_completed(futures):
                idx = futures[future]

                try:
                    results[idx] = future.result()
                except Exception as exc:
                    logger.exception(
                        "Parallel preprocessing failed",
                        extra={"index": idx},
                    )
                    raise RuntimeError("Parallel preprocessing failed") from exc

        # All entries in `results` are populated when we reach this point:
        # any worker failure raises RuntimeError above before we return,
        # so the `if res is not None` filter is dead code that would silently
        # shorten the output list and break positional correspondence with the
        # input `texts`.
        return results  # type: ignore[return-value]

    @staticmethod
    def _process_text_static(text: str) -> PreprocessingResult:
        """
        Static helper used for parallel preprocessing.

        A lightweight pipeline is instantiated per worker.
        """

        try:
            nlp = _get_nlp()

            normalized_text = text.strip()
            normalized_text = re.sub(r"\s+", " ", normalized_text)
            normalized_text = re.sub(r"[^\w\s\.\,\!\?\-']", "", normalized_text)

            try:
                language = detect(text)
            except LangDetectException:
                language = "unknown"

            doc = nlp(normalized_text)

            tokens = [t.text.lower() for t in doc if not t.is_space]
            lemmas = [t.lemma_.lower() for t in doc if not t.is_space]
            sentences = [s.text.strip() for s in doc.sents if s.text.strip()]

            return PreprocessingResult(
                normalized_text=normalized_text,
                tokens=tokens,
                lemmas=lemmas,
                sentences=sentences,
                language=language,
            )

        except Exception as exc:
            raise RuntimeError("Worker preprocessing failed") from exc

    def _detect_language(self, text: str) -> str:
        """
        Detect language of input text.

        Parameters
        ----------
        text : str

        Returns
        -------
        str
        """

        try:
            return detect(text)
        except LangDetectException:
            logger.warning("Language detection failed")
            return "unknown"

    def _normalize_text(self, text: str) -> str:
        """
        Normalize text by removing excessive whitespace and unwanted characters.
        """

        text = text.strip()

        text = re.sub(r"\s+", " ", text)

        text = re.sub(r"[^\w\s\.\,\!\?\-']", "", text)

        return text

    def _extract_tokens(self, doc) -> List[str]:
        """
        Extract tokens from spaCy document.
        """

        return [token.text.lower() for token in doc if not token.is_space]

    def _extract_lemmas(self, doc) -> List[str]:
        """
        Extract lemmas from spaCy document.
        """

        return [token.lemma_.lower() for token in doc if not token.is_space]

    def _extract_sentences(self, doc) -> List[str]:
        """
        Extract sentence list from spaCy document.
        """

        return [sent.text.strip() for sent in doc.sents if sent.text.strip()]
