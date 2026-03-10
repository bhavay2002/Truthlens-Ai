"""
Metadata Feature Engineering for TruthLens AI
Extracts structural and statistical features from news articles
"""

import pandas as pd
import logging
import re

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# -------------------------------------------------
# Helper Functions
# -------------------------------------------------

def count_words(text):
    """Count words in text, handling None/empty values"""
    if text is None or pd.isna(text):
        return 0
    return len(str(text).split())


def count_sentences(text):
    """Count sentences in text, handling None/empty values"""
    if text is None or pd.isna(text):
        return 0
    return max(0, len(re.split(r"[.!?]+", str(text))) - 1)


def count_exclamations(text):
    """Count exclamation marks, handling None/empty values"""
    if text is None or pd.isna(text):
        return 0
    return str(text).count("!")


def count_questions(text):
    """Count question marks, handling None/empty values"""
    if text is None or pd.isna(text):
        return 0
    return str(text).count("?")


def uppercase_ratio(text):
    """Calculate ratio of uppercase characters, handling None/empty values"""
    if text is None or pd.isna(text):
        return 0.0
    text = str(text)
    if len(text) == 0:
        return 0.0
    return sum(1 for c in text if c.isupper()) / len(text)


# -------------------------------------------------
# Feature Extraction
# -------------------------------------------------

def extract_metadata_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract metadata features from dataset

    Features include:
    - title length
    - text length
    - word counts
    - sentence counts
    - punctuation signals
    - capitalization ratio
    - author presence
    """

    try:

        logger.info("Extracting metadata features...")

        # -------------------------------------------------
        # Title Features
        # -------------------------------------------------

        if "title" in df.columns:

            df["title_length"] = df["title"].astype(str).apply(len)
            df["title_word_count"] = df["title"].apply(count_words)
            df["title_uppercase_ratio"] = df["title"].apply(uppercase_ratio)

        # -------------------------------------------------
        # Text Features
        # -------------------------------------------------

        df["text_length"] = df["text"].astype(str).apply(len)
        df["word_count"] = df["text"].apply(count_words)
        df["sentence_count"] = df["text"].apply(count_sentences)

        # -------------------------------------------------
        # Punctuation Signals
        # -------------------------------------------------

        df["exclamation_count"] = df["text"].apply(count_exclamations)
        df["question_count"] = df["text"].apply(count_questions)

        # -------------------------------------------------
        # Capitalization
        # -------------------------------------------------

        df["uppercase_ratio"] = df["text"].apply(uppercase_ratio)

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

        logger.info("Metadata feature extraction completed")

        return df

    except Exception as e:

        logger.error(f"Metadata feature extraction failed: {e}")
        raise