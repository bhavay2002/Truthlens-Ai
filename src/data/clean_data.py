"""
File: src/data/clean_data.py

Purpose
-------
Research-grade dataset cleaning and text normalization utilities.

Designed for large NLP pipelines and multi-dataset systems such as
multi-task transformer training.

Features
--------
- Robust text normalization
- Vectorized dataset cleaning
- Configurable preprocessing pipeline
- Dataset diagnostics
- Parallel processing support
- Multi-dataset schema compatibility
"""

from __future__ import annotations

import logging
import re
import unicodedata
from typing import Optional, List

import pandas as pd

logger = logging.getLogger(__name__)


# -------------------------------------------------
# Regex Patterns (Precompiled for performance)
# -------------------------------------------------

URL_PATTERN = re.compile(r"http\S+|www\S+|https\S+")
EMAIL_PATTERN = re.compile(r"\S+@\S+")
MENTION_PATTERN = re.compile(r"@\w+|#\w+")
HTML_PATTERN = re.compile(r"<.*?>")
WHITESPACE_PATTERN = re.compile(r"\s+")
NUMBER_PATTERN = re.compile(r"\d+")
REPEATED_CHARS = re.compile(r"(.)\1{2,}")

EMOJI_PATTERN = re.compile(
    "["
    "\U0001F600-\U0001F64F"
    "\U0001F300-\U0001F5FF"
    "\U0001F680-\U0001F6FF"
    "\U0001F900-\U0001F9FF"
    "\U0001FA00-\U0001FAFF"
    "]+",
    flags=re.UNICODE,
)


# -------------------------------------------------
# Basic Normalization Utilities
# -------------------------------------------------

def normalize_unicode(text: str) -> str:

    text = unicodedata.normalize("NFKD", text)

    replacements = {
        "“": '"',
        "”": '"',
        "’": "'",
        "‘": "'",
    }

    for old, new in replacements.items():
        text = text.replace(old, new)

    return text


def remove_emojis(text: str) -> str:

    return EMOJI_PATTERN.sub("", text)


def normalize_numbers(text: str) -> str:

    return NUMBER_PATTERN.sub("<NUM>", text)


def normalize_repeated_chars(text: str) -> str:

    return REPEATED_CHARS.sub(r"\1\1", text)


def expand_contractions(text: str) -> str:

    try:
        import contractions
        return contractions.fix(text)
    except ImportError:
        return text


# -------------------------------------------------
# Core Text Cleaning
# -------------------------------------------------

def clean_text(
    text: str,
    normalize_nums: bool = True,
    remove_urls: bool = True,
    remove_html: bool = True,
) -> str:

    if text is None:
        return ""

    text = str(text)

    text = normalize_unicode(text)
    text = remove_emojis(text)

    text = text.lower()

    text = expand_contractions(text)

    if remove_urls:
        text = URL_PATTERN.sub("", text)
        text = EMAIL_PATTERN.sub("", text)

    text = MENTION_PATTERN.sub("", text)

    if remove_html:
        text = HTML_PATTERN.sub("", text)

    text = normalize_repeated_chars(text)

    if normalize_nums:
        text = normalize_numbers(text)

    text = re.sub(r"[!?]{2,}", "!", text)
    text = re.sub(r"[.]{2,}", ".", text)

    text = re.sub(r"[^a-zA-Z0-9\s.,!?<>]", "", text)

    text = WHITESPACE_PATTERN.sub(" ", text).strip()

    return text


# -------------------------------------------------
# Dataset Diagnostics
# -------------------------------------------------

def dataset_text_statistics(
    df: pd.DataFrame,
    text_column: str,
) -> dict:

    if text_column not in df.columns:
        raise ValueError(f"{text_column} not found")

    lengths = df[text_column].astype(str).apply(len)

    stats = {
        "rows": len(df),
        "avg_length": lengths.mean(),
        "max_length": lengths.max(),
        "min_length": lengths.min(),
    }

    logger.info("Dataset text statistics: %s", stats)

    return stats


# -------------------------------------------------
# DataFrame Cleaning Pipeline
# -------------------------------------------------

def clean_dataframe(
    df: pd.DataFrame,
    text_column: str = "text",
    title_column: Optional[str] = None,
    min_words: int = 20,
) -> pd.DataFrame:

    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame")

    if text_column not in df.columns:
        raise ValueError(f"Column '{text_column}' not found")

    df = df.copy()

    initial_rows = len(df)

    logger.info("Initial dataset size: %d", initial_rows)

    if initial_rows == 0:
        return df

    # Merge title + body text
    if title_column and title_column in df.columns:

        df[text_column] = (
            df[title_column].fillna("")
            + " </s> "
            + df[text_column].fillna("")
        )

    # Remove duplicates
    df = df.drop_duplicates(subset=[text_column])

    # Remove missing values
    df = df.dropna(subset=[text_column])

    # Convert to string
    df[text_column] = df[text_column].astype(str)

    # Apply cleaning
    df[text_column] = df[text_column].map(clean_text)

    # Word filtering
    word_counts = df[text_column].apply(lambda x: len(x.split()))

    df = df[word_counts >= min_words]

    df = df.reset_index(drop=True)

    final_rows = len(df)

    logger.info("Final dataset size: %d", final_rows)
    logger.info("Rows removed: %d", initial_rows - final_rows)

    return df


# -------------------------------------------------
# Parallel Cleaning (Large Datasets)
# -------------------------------------------------

def clean_dataframe_parallel(
    df: pd.DataFrame,
    text_column: str = "text",
    workers: int = 4,
) -> pd.DataFrame:

    try:

        from multiprocessing import Pool

        texts = df[text_column].astype(str).tolist()

        with Pool(workers) as p:

            cleaned = p.map(clean_text, texts)

        df = df.copy()
        df[text_column] = cleaned

        return df

    except Exception as e:

        logger.warning(
            "Parallel cleaning failed, falling back to single process: %s",
            e,
        )

        df[text_column] = df[text_column].map(clean_text)

        return df


# -------------------------------------------------
# Optional NLP Processing
# -------------------------------------------------

def advanced_text_preprocessing(
    text: str,
    remove_stopwords: bool = False,
    lemmatize: bool = False,
) -> str:

    text = clean_text(text)

    try:

        import nltk
        from nltk.corpus import stopwords
        from nltk.stem import WordNetLemmatizer

        nltk.download("stopwords", quiet=True)
        nltk.download("wordnet", quiet=True)

        words = text.split()

        if remove_stopwords:

            stop_words = set(stopwords.words("english"))
            words = [w for w in words if w not in stop_words]

        if lemmatize:

            lemmatizer = WordNetLemmatizer()
            words = [lemmatizer.lemmatize(w) for w in words]

        text = " ".join(words)

    except ImportError:

        logger.warning("NLTK not installed. Skipping NLP preprocessing.")

    return text