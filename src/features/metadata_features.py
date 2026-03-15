"""
File: metadata_features.py

Purpose
-------
Metadata feature engineering module for TruthLens AI.

This module extracts structural and stylistic features from news
articles that can help identify misinformation patterns.

Input
-----
df : pandas.DataFrame

Required columns:
    text : str

Optional columns:
    title : str
    author : str
    source : str

Output
------
pandas.DataFrame
    Original dataframe with additional metadata features.
"""

import logging
import re
from typing import Optional

import pandas as pd

from src.utils.input_validation import ensure_dataframe


# ---------------------------------------------------------
# Logging Configuration
# ---------------------------------------------------------

logger = logging.getLogger(__name__)


# ---------------------------------------------------------
# Helper Functions
# ---------------------------------------------------------

def _count_words(text: Optional[str]) -> int:
    """
    Count words in text safely.
    """

    if text is None or pd.isna(text):
        return 0

    return len(str(text).split())


def _count_sentences(text: Optional[str]) -> int:
    """
    Estimate number of sentences.
    """

    if text is None or pd.isna(text):
        return 0

    sentences = re.split(r"[.!?]+", str(text))

    return max(0, len(sentences) - 1)


def _count_exclamations(text: Optional[str]) -> int:
    """
    Count exclamation marks.
    """

    if text is None or pd.isna(text):
        return 0

    return str(text).count("!")


def _count_questions(text: Optional[str]) -> int:
    """
    Count question marks.
    """

    if text is None or pd.isna(text):
        return 0

    return str(text).count("?")


def _uppercase_ratio(text: Optional[str]) -> float:
    """
    Compute ratio of uppercase characters.
    """

    if text is None or pd.isna(text):
        return 0.0

    text = str(text)

    if len(text) == 0:
        return 0.0

    uppercase_chars = sum(1 for c in text if c.isupper())

    return uppercase_chars / len(text)


# ---------------------------------------------------------
# Feature Extraction Pipeline
# ---------------------------------------------------------

def extract_metadata_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract metadata features from news dataset.

    Parameters
    ----------
    df : pandas.DataFrame
        Input dataset containing article text and optional metadata.

    Returns
    -------
    pandas.DataFrame
        Dataset with additional engineered features.
    """

    try:

        ensure_dataframe(df, name="df", required_columns=["text"])

        logger.info("Starting metadata feature extraction")

        # -------------------------------------------------
        # Title Features
        # -------------------------------------------------

        if "title" in df.columns:

            df["title_length"] = df["title"].astype(str).apply(len)

            df["title_word_count"] = df["title"].apply(_count_words)

            df["title_uppercase_ratio"] = df["title"].apply(_uppercase_ratio)

        # -------------------------------------------------
        # Text Features
        # -------------------------------------------------

        df["text_length"] = df["text"].astype(str).apply(len)

        df["word_count"] = df["text"].apply(_count_words)

        df["sentence_count"] = df["text"].apply(_count_sentences)

        # -------------------------------------------------
        # Punctuation Signals
        # -------------------------------------------------

        df["exclamation_count"] = df["text"].apply(_count_exclamations)

        df["question_count"] = df["text"].apply(_count_questions)

        # -------------------------------------------------
        # Capitalization Signals
        # -------------------------------------------------

        df["uppercase_ratio"] = df["text"].apply(_uppercase_ratio)

        # -------------------------------------------------
        # Author Features
        # -------------------------------------------------

        if "author" in df.columns:

            df["has_author"] = df["author"].notnull().astype(int)

            df["author_name_length"] = df["author"].astype(str).apply(len)

        # -------------------------------------------------
        # Source Features
        # -------------------------------------------------

        if "source" in df.columns:

            df["source_length"] = df["source"].astype(str).apply(len)

        logger.info("Metadata feature extraction completed successfully")

        return df

    except Exception as e:

        logger.exception("Metadata feature extraction failed")

        raise