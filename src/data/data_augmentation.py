"""
File: src/data/data_augmentation.py

Purpose
-------
Research-grade NLP data augmentation utilities.

Supports:
- synonym replacement
- random deletion
- random swap
- sentence shuffle
- class-aware augmentation
- configurable augmentation pipeline
- parallel dataset augmentation

Designed for transformer training pipelines
(e.g., RoBERTa / DeBERTa / BERT).

Dependencies
------------
pandas
random
nltk
src.utils.input_validation
"""

from __future__ import annotations

import logging
import random
from multiprocessing import Pool
from typing import Callable, List

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

random.seed(42)


# ------------------------------------------------
# Synonym Lookup
# ------------------------------------------------

def get_synonyms(word: str) -> List[str]:

    synonyms = set()

    try:
        synsets = wordnet.synsets(word)
    except LookupError:
        return []

    for syn in synsets:

        for lemma in syn.lemmas():

            synonym = lemma.name().replace("_", " ").lower()

            if synonym != word:
                synonyms.add(synonym)

    return list(synonyms)


# ------------------------------------------------
# Augmentation Operations
# ------------------------------------------------

def synonym_replacement(text: str, n: int = 2) -> str:

    words = str(text).split()
    candidates = [w for w in words if w not in STOPWORDS and len(w) > 3]

    random.shuffle(candidates)

    replaced = 0

    for word in candidates:

        synonyms = get_synonyms(word)

        if synonyms:

            synonym = random.choice(synonyms)

            words = [synonym if w == word else w for w in words]

            replaced += 1

        if replaced >= n:
            break

    return " ".join(words)


def random_deletion(text: str, p: float = 0.1) -> str:

    words = str(text).split()

    if len(words) <= 5:
        return text

    new_words = [w for w in words if random.random() > p]

    if not new_words:
        return random.choice(words)

    return " ".join(new_words)


def random_swap(text: str) -> str:

    words = str(text).split()

    if len(words) < 3:
        return text

    i1, i2 = random.sample(range(len(words)), 2)

    words[i1], words[i2] = words[i2], words[i1]

    return " ".join(words)


def sentence_shuffle(text: str) -> str:

    sentences = text.split(".")

    if len(sentences) < 2:
        return text

    random.shuffle(sentences)

    return ".".join(sentences)


# ------------------------------------------------
# Augmentation Pipeline
# ------------------------------------------------

AUGMENTATION_OPERATIONS: List[Callable[[str], str]] = [
    synonym_replacement,
    random_deletion,
    random_swap,
    sentence_shuffle,
]


def augment_text(text: str) -> str:

    text = str(text).strip()

    if not text:
        return ""

    operation = random.choice(AUGMENTATION_OPERATIONS)

    return operation(text)


# ------------------------------------------------
# Row-Level Augmentation
# ------------------------------------------------

def _augment_row(row, text_column):

    text = str(row[text_column])

    new_row = row.copy()

    new_row[text_column] = augment_text(text)

    return new_row


# ------------------------------------------------
# Dataset Augmentation
# ------------------------------------------------

def augment_dataset(
    df: pd.DataFrame,
    text_column: str = "text",
    multiplier: int = 2,
) -> pd.DataFrame:

    ensure_dataframe(df, name="df", required_columns=[text_column], min_rows=1)

    ensure_non_empty_text_column(df, text_column, name="df")

    ensure_positive_int(multiplier, name="multiplier", min_value=1)

    if multiplier <= 1:

        logger.info("Multiplier <=1, returning original dataset")

        return df.copy()

    augmented_rows = []

    records = df.to_dict("records")

    for row in records:

        for _ in range(multiplier - 1):

            augmented_rows.append(_augment_row(row, text_column))

    augmented_df = pd.concat(
        [df, pd.DataFrame(augmented_rows)],
        ignore_index=True,
    )

    logger.info(
        "Dataset augmentation complete | original=%d augmented=%d total=%d",
        len(df),
        len(augmented_rows),
        len(augmented_df),
    )

    return augmented_df


# ------------------------------------------------
# Parallel Augmentation
# ------------------------------------------------

def augment_dataset_parallel(
    df: pd.DataFrame,
    text_column: str = "text",
    multiplier: int = 2,
    workers: int = 4,
) -> pd.DataFrame:

    ensure_dataframe(df, name="df", required_columns=[text_column], min_rows=1)

    records = df.to_dict("records")

    tasks = []

    for row in records:

        for _ in range(multiplier - 1):

            tasks.append((row, text_column))

    with Pool(workers) as pool:

        augmented_rows = pool.starmap(_augment_row, tasks)

    augmented_df = pd.concat(
        [df, pd.DataFrame(augmented_rows)],
        ignore_index=True,
    )

    logger.info(
        "Parallel augmentation complete | total=%d",
        len(augmented_df),
    )

    return augmented_df