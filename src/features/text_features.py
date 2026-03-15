"""
File: text_features.py

Purpose
-------
Text feature engineering module for TruthLens AI.

This module builds TF-IDF vector representations for text data.
TF-IDF features are commonly used for classical machine learning
models and baseline comparisons against transformer architectures.

Input
-----
texts : List[str]

Output
------
csr_matrix
    Sparse matrix of TF-IDF features

TfidfVectorizer
    Trained vectorizer instance
"""

import logging
from typing import List, Tuple

from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer

from src.utils.input_validation import ensure_non_empty_text_list

# ---------------------------------------------------------
# Logging Configuration
# ---------------------------------------------------------

logger = logging.getLogger(__name__)


# ---------------------------------------------------------
# TF-IDF Vectorizer Builder
# ---------------------------------------------------------


def build_tfidf_vectorizer(
    max_features: int = 5000,
    ngram_range: Tuple[int, int] = (1, 2),
    min_df: int = 1,
    max_df: float = 0.9,
) -> TfidfVectorizer:
    """
    Create TF-IDF vectorizer with recommended defaults.

    Parameters
    ----------
    max_features : int
        Maximum vocabulary size.

    ngram_range : tuple
        N-gram range used for feature extraction.

    min_df : int
        Minimum document frequency threshold.

    max_df : float
        Maximum document frequency threshold.

    Returns
    -------
    TfidfVectorizer
    """

    vectorizer = TfidfVectorizer(
        max_features=max_features,
        stop_words="english",
        ngram_range=ngram_range,
        min_df=min_df,
        max_df=max_df,
        sublinear_tf=True,
    )

    logger.info(
        "TF-IDF vectorizer created | max_features=%d | ngram_range=%s",
        max_features,
        ngram_range,
    )

    return vectorizer


# ---------------------------------------------------------
# Fit TF-IDF Model
# ---------------------------------------------------------


def tfidf_fit_transform(
    texts: List[str],
    vectorizer: TfidfVectorizer | None = None,
) -> Tuple[csr_matrix, TfidfVectorizer]:
    """
    Fit TF-IDF vectorizer and transform text corpus.

    Parameters
    ----------
    texts : List[str]
        Training corpus.

    vectorizer : TfidfVectorizer | None
        Optional preconfigured vectorizer.

    Returns
    -------
    csr_matrix
        Sparse TF-IDF feature matrix.

    TfidfVectorizer
        Trained vectorizer instance.
    """

    ensure_non_empty_text_list(texts)

    if vectorizer is None:
        vectorizer = build_tfidf_vectorizer()

    logger.info("Fitting TF-IDF vectorizer")

    X = vectorizer.fit_transform(texts)

    logger.info(
        "TF-IDF matrix generated | shape=%s | vocabulary=%d",
        X.shape,
        len(vectorizer.vocabulary_),
    )

    return X, vectorizer


# ---------------------------------------------------------
# Transform New Text
# ---------------------------------------------------------


def tfidf_transform(
    texts: List[str],
    vectorizer: TfidfVectorizer,
) -> csr_matrix:
    """
    Transform new texts using a trained vectorizer.

    Parameters
    ----------
    texts : List[str]

    vectorizer : TfidfVectorizer

    Returns
    -------
    csr_matrix
    """

    ensure_non_empty_text_list(texts)

    logger.info("Transforming texts using trained TF-IDF vectorizer")

    X = vectorizer.transform(texts)

    return X


# ---------------------------------------------------------
# Vocabulary Extraction
# ---------------------------------------------------------


def get_feature_names(vectorizer: TfidfVectorizer) -> List[str]:
    """
    Return vocabulary terms from TF-IDF vectorizer.
    """

    return vectorizer.get_feature_names_out().tolist()


# ---------------------------------------------------------
# Backward Compatibility Wrapper
# ---------------------------------------------------------


def tfidf_features(
    texts: List[str],
    max_features: int = 5000,
) -> Tuple[csr_matrix, TfidfVectorizer]:
    """
    Backward-compatible wrapper used by legacy code/tests.
    """

    vectorizer = build_tfidf_vectorizer(max_features=max_features)

    return tfidf_fit_transform(texts, vectorizer=vectorizer)
