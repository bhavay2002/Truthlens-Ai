"""
File: src/data/clean_data.py

Purpose
-------
Provides text normalization and dataset cleaning utilities for NLP pipelines.
Removes noise such as URLs, HTML tags, emojis, mentions, and repeated characters.
Also supports optional NLP preprocessing like stopword removal and lemmatization.

Typical Usage
-------------
Used during dataset preprocessing before model training (e.g., RoBERTa classifier).

Inputs
------
text : str
    Raw text data (news article, tweet, etc.)

df : pandas.DataFrame
    Dataset containing text column.

Outputs
-------
clean_text(text) -> str
clean_dataframe(df) -> pandas.DataFrame
advanced_text_preprocessing(text) -> str

Dependencies
------------
pandas
re
logging
unicodedata
optional: contractions
optional: nltk
"""

from __future__ import annotations

import logging
import re
import unicodedata
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)


# -------------------------------------------------
# Precompiled Regex Patterns (Performance optimized)
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
    "\U0001F700-\U0001F77F"
    "\U0001F900-\U0001F9FF"
    "\U0001FA00-\U0001FAFF"
    "]+",
    flags=re.UNICODE,
)


# -------------------------------------------------
# Text Normalization Utilities
# -------------------------------------------------

def normalize_unicode(text: str) -> str:
    """
    Normalize unicode characters and quotation marks.

    Parameters
    ----------
    text : str
        Input raw text.

    Returns
    -------
    str
        Normalized text.
    """

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


def expand_contractions(text: str) -> str:
    """
    Expand English contractions if library available.

    Example:
    don't -> do not
    """

    try:
        import contractions

        return contractions.fix(text)

    except ImportError:
        return text


def normalize_numbers(text: str) -> str:
    """
    Replace numeric values with <NUM> token.
    """

    return NUMBER_PATTERN.sub("<NUM>", text)


def remove_emojis(text: str) -> str:
    """
    Remove emojis from text.
    """

    return EMOJI_PATTERN.sub("", text)


def normalize_repeated_chars(text: str) -> str:
    """
    Normalize repeated characters.

    Example:
    soooo -> soo
    """

    return REPEATED_CHARS.sub(r"\1\1", text)


# -------------------------------------------------
# Core Cleaning Function
# -------------------------------------------------

def clean_text(text: str, normalize_nums: bool = True) -> str:
    """
    Clean and normalize input text.

    Steps
    -----
    - Unicode normalization
    - Remove emojis
    - Lowercase conversion
    - Expand contractions
    - Remove URLs / emails / mentions / HTML
    - Normalize repeated characters
    - Replace numbers
    - Remove noisy symbols

    Parameters
    ----------
    text : str
        Raw input text.

    normalize_nums : bool
        Whether to replace numbers with <NUM> token.

    Returns
    -------
    str
        Clean normalized text.
    """

    original_text = "" if text is None else text
    text = str(text)

    text = normalize_unicode(text)
    text = remove_emojis(text)

    text = text.lower()
    text = expand_contractions(text)

    text = URL_PATTERN.sub("", text)
    text = EMAIL_PATTERN.sub("", text)
    text = MENTION_PATTERN.sub("", text)
    text = HTML_PATTERN.sub("", text)

    text = normalize_repeated_chars(text)

    if normalize_nums:
        text = normalize_numbers(text)

    text = re.sub(r"[!?]{2,}", "!", text)
    text = re.sub(r"[.]{2,}", ".", text)

    text = re.sub(r"[^a-zA-Z0-9\s.,!?<>]", "", text)

    text = WHITESPACE_PATTERN.sub(" ", text).strip()

    # Fallback safeguard
    if not text or len(text) < 3:
        return str(original_text).lower().strip()

    return text


# -------------------------------------------------
# DataFrame Cleaning Pipeline
# -------------------------------------------------

def clean_dataframe(
    df: pd.DataFrame,
    text_column: str = "text",
    title_column: Optional[str] = None,
    min_len: int = 30,
) -> pd.DataFrame:
    """
    Clean dataset containing text data.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataset.

    text_column : str
        Column containing main article text.

    title_column : Optional[str]
        Optional column containing article title.

    min_len : int
        Minimum word count threshold.

    Returns
    -------
    pd.DataFrame
        Cleaned dataset.
    """

    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame")

    if min_len < 0:
        raise ValueError("min_len must be >= 0")

    if text_column not in df.columns:
        raise ValueError(f"Column '{text_column}' not found")

    df = df.copy()

    initial_rows = len(df)
    logger.info(f"Initial dataset size: {initial_rows}")

    if initial_rows == 0:
        logger.warning("Input dataframe is empty; returning empty dataframe")
        return df

    # Merge title + text
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

    df[text_column] = df[text_column].astype(str).str.strip()

    # Remove empty rows
    df = df[df[text_column].str.len() > 0]

    # Apply text cleaning
    df[text_column] = df[text_column].apply(clean_text)

    # Remove short texts
    df["word_count"] = df[text_column].apply(lambda x: len(str(x).split()))

    df = df[df["word_count"] >= min_len]

    df = df.drop(columns=["word_count"])

    df = df.reset_index(drop=True)

    final_rows = len(df)

    logger.info(f"Final dataset size: {final_rows}")
    logger.info(f"Rows removed: {initial_rows - final_rows}")

    retention = (final_rows / initial_rows) * 100 if initial_rows else 0
    logger.info(f"Retention rate: {retention:.2f}%")

    return df


# -------------------------------------------------
# Optional NLP Preprocessing
# -------------------------------------------------

def advanced_text_preprocessing(
    text: str,
    remove_stopwords: bool = False,
    lemmatize: bool = False,
) -> str:
    """
    Advanced NLP preprocessing.

    Optional:
    - Stopword removal
    - Lemmatization

    Parameters
    ----------
    text : str
        Input text.

    remove_stopwords : bool
        Whether to remove stopwords.

    lemmatize : bool
        Whether to apply lemmatization.

    Returns
    -------
    str
        Preprocessed text.
    """

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
        logger.warning(
            "NLTK not installed, skipping advanced preprocessing"
        )

    return text
