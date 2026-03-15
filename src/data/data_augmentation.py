"""
File: src/data/data_augmentation.py

Purpose
-------
Provide NLP data augmentation utilities to increase dataset diversity.
Augmentation techniques include synonym replacement, random deletion,
and random word swapping.

This module is typically used during dataset preparation for training
transformer-based models such as RoBERTa.

Inputs
------
text : str
    Input text sample.

df : pandas.DataFrame
    Dataset containing text column.

Outputs
-------
augment_text(text) -> str
augment_dataset(df) -> pandas.DataFrame

Dependencies
------------
pandas
random
nltk
wordnet
stopwords
src.utils.input_validation
"""

from __future__ import annotations

import logging
import random
from typing import List

import pandas as pd
import nltk
from nltk.corpus import wordnet, stopwords

from src.utils.input_validation import (
    ensure_dataframe,
    ensure_non_empty_text_column,
    ensure_positive_int,
)

logger = logging.getLogger(__name__)

# ------------------------------------------------
# NLTK Setup
# ------------------------------------------------

try:
    nltk.download("wordnet", quiet=True)
    nltk.download("stopwords", quiet=True)

    STOPWORDS = set(stopwords.words("english"))

except Exception as e:

    logger.warning(
        "Failed to download NLTK resources: %s. Using empty stopword set.",
        e,
    )

    STOPWORDS = set()

# Ensure deterministic augmentation for reproducibility
random.seed(42)


# ------------------------------------------------
# Synonym Extraction
# ------------------------------------------------

def get_synonyms(word: str) -> List[str]:
    """
    Retrieve synonyms for a word using WordNet.

    Parameters
    ----------
    word : str
        Input word.

    Returns
    -------
    List[str]
        List of synonym candidates.
    """

    synonyms = set()

    try:
        synsets = wordnet.synsets(word)
    except LookupError:
        logger.warning("WordNet resource unavailable; synonym replacement disabled")
        return []

    for syn in synsets:

        for lemma in syn.lemmas():

            synonym = lemma.name().replace("_", " ").lower()

            if synonym != word:
                synonyms.add(synonym)

    return list(synonyms)


# ------------------------------------------------
# Synonym Replacement
# ------------------------------------------------

def synonym_replacement(text: str, n: int = 2) -> str:
    """
    Replace up to n words with synonyms.

    Parameters
    ----------
    text : str
        Input text.

    n : int
        Maximum number of replacements.

    Returns
    -------
    str
        Augmented text.
    """

    words = str(text).split()

    new_words = words.copy()

    candidates = [
        word
        for word in words
        if word not in STOPWORDS and len(word) > 3
    ]

    random.shuffle(candidates)

    replaced = 0

    for word in candidates:

        synonyms = get_synonyms(word)

        if synonyms:

            synonym = random.choice(synonyms)

            new_words = [
                synonym if w == word else w
                for w in new_words
            ]

            replaced += 1

        if replaced >= n:
            break

    return " ".join(new_words)


# ------------------------------------------------
# Random Deletion
# ------------------------------------------------

def random_deletion(text: str, p: float = 0.1) -> str:
    """
    Randomly remove words from text.

    Parameters
    ----------
    text : str
        Input text.

    p : float
        Probability of removing each word.

    Returns
    -------
    str
        Augmented text.
    """

    words = str(text).split()

    if len(words) <= 5:
        return text

    new_words = [
        word for word in words
        if random.random() > p
    ]

    if not new_words:
        return random.choice(words)

    return " ".join(new_words)


# ------------------------------------------------
# Random Word Swap
# ------------------------------------------------

def random_swap(text: str) -> str:
    """
    Swap two random words in text.

    Parameters
    ----------
    text : str
        Input text.

    Returns
    -------
    str
        Augmented text.
    """

    words = str(text).split()

    if len(words) < 3:
        return text

    idx1 = random.randint(0, len(words) - 1)
    idx2 = random.randint(0, len(words) - 1)

    words[idx1], words[idx2] = words[idx2], words[idx1]

    return " ".join(words)


# ------------------------------------------------
# Single Text Augmentation
# ------------------------------------------------

def augment_text(text: str) -> str:
    """
    Apply a random augmentation operation.

    Operations include:
    - synonym replacement
    - random deletion
    - random swap

    Parameters
    ----------
    text : str
        Input text.

    Returns
    -------
    str
        Augmented text.
    """

    text = str(text).strip()
    if not text:
        return ""

    operations = [
        synonym_replacement,
        random_deletion,
        random_swap,
    ]

    operation = random.choice(operations)

    return operation(text)


# ------------------------------------------------
# Dataset Augmentation
# ------------------------------------------------

def augment_dataset(
    df: pd.DataFrame,
    text_column: str = "text",
    multiplier: int = 2,
) -> pd.DataFrame:
    """
    Augment dataset to increase training samples.

    Parameters
    ----------
    df : pandas.DataFrame
        Input dataset.

    text_column : str
        Column containing text data.

    multiplier : int
        Dataset expansion factor.

        multiplier=1 -> no augmentation
        multiplier=2 -> dataset roughly doubled

    Returns
    -------
    pandas.DataFrame
        Augmented dataset.
    """

    ensure_dataframe(df, name="df", required_columns=[text_column], min_rows=1)

    ensure_non_empty_text_column(df, text_column, name="df")

    ensure_positive_int(multiplier, name="multiplier", min_value=1)

    if multiplier == 1:

        logger.info("Augmentation multiplier <= 1. Returning original dataset.")

        return df.copy()

    augmented_rows = []

    for _, row in df.iterrows():

        text = str(row[text_column])

        for _ in range(multiplier - 1):

            augmented_text = augment_text(text)

            new_row = row.copy()

            new_row[text_column] = augmented_text

            augmented_rows.append(new_row)

    augmented_df = pd.concat(
        [df, pd.DataFrame(augmented_rows)],
        ignore_index=True,
    )

    logger.info(
        "Augmentation complete: original=%s, augmented=%s, total=%s",
        len(df),
        len(augmented_rows),
        len(augmented_df),
    )

    return augmented_df
