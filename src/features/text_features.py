"""
Text Feature Engineering Module
Creates TF-IDF features for machine learning models
"""

import logging
from typing import Tuple, List
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import csr_matrix

logger = logging.getLogger(__name__)


# -------------------------------------------------
# TF-IDF Feature Builder
# -------------------------------------------------

def build_tfidf_vectorizer(
    max_features: int = 5000,
    ngram_range: Tuple[int, int] = (1, 2),
    min_df: int = 1,
    max_df: float = 0.9
) -> TfidfVectorizer:
    """
    Create TF-IDF vectorizer with professional defaults
    """

    vectorizer = TfidfVectorizer(
        max_features=max_features,
        stop_words="english",
        ngram_range=ngram_range,
        min_df=min_df,
        max_df=max_df,
        sublinear_tf=True
    )

    logger.info(
        f"TF-IDF vectorizer created | max_features={max_features}, "
        f"ngram_range={ngram_range}"
    )

    return vectorizer


# -------------------------------------------------
# Train TF-IDF
# -------------------------------------------------

def tfidf_fit_transform(
    texts: List[str],
    vectorizer: TfidfVectorizer = None
) -> Tuple[csr_matrix, TfidfVectorizer]:
    """
    Fit TF-IDF vectorizer and transform texts

    Returns:
        sparse matrix features
        trained vectorizer
    """

    if vectorizer is None:
        vectorizer = build_tfidf_vectorizer()

    logger.info("Fitting TF-IDF vectorizer...")

    X = vectorizer.fit_transform(texts)

    logger.info(
        f"TF-IDF matrix shape: {X.shape} | vocabulary size: {len(vectorizer.vocabulary_)}"
    )

    return X, vectorizer


# -------------------------------------------------
# Transform New Data
# -------------------------------------------------

def tfidf_transform(
    texts: List[str],
    vectorizer: TfidfVectorizer
) -> csr_matrix:
    """
    Transform new texts using trained vectorizer
    """

    logger.info("Transforming texts using existing TF-IDF vectorizer...")

    X = vectorizer.transform(texts)

    return X


# -------------------------------------------------
# Feature Names
# -------------------------------------------------

def get_feature_names(vectorizer: TfidfVectorizer) -> List[str]:
    """
    Get TF-IDF vocabulary terms
    """

    return vectorizer.get_feature_names_out().tolist()


# -------------------------------------------------
# Backward Compatibility
# -------------------------------------------------

def tfidf_features(
    texts: List[str],
    max_features: int = 5000
) -> Tuple[csr_matrix, TfidfVectorizer]:
    """
    Backward-compatible wrapper used by existing code/tests.
    """
    vectorizer = build_tfidf_vectorizer(max_features=max_features)
    return tfidf_fit_transform(texts, vectorizer=vectorizer)
