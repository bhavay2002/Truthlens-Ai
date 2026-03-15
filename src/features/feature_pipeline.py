"""
File: feature_pipeline.py

Purpose
-------
Unified feature engineering pipeline for TruthLens AI.

This module combines multiple feature sources into a single
tokenized representation appended to the original text.

Features integrated:
- TF-IDF keywords
- Metadata signals
- Source credibility
- Bias lexicon features
- Emotion lexicon features

Input
-----
train_df : pandas.DataFrame

Output
------
Tuple[pandas.DataFrame, TfidfVectorizer]
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Tuple, Any

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

from src.features.bias.bias_lexicon import compute_bias_features
from src.features.emotion.emotion_lexicon import EmotionLexiconAnalyzer
from src.features.metadata_features import extract_metadata_features
from src.features.source_features import add_source_features
from src.features.text_features import (
    build_tfidf_vectorizer,
    get_feature_names,
    tfidf_fit_transform,
    tfidf_transform,
)

from src.utils.input_validation import (
    ensure_dataframe,
    ensure_non_empty_text_column,
    ensure_positive_int,
)


# ---------------------------------------------------------
# Logging Configuration
# ---------------------------------------------------------

logger = logging.getLogger(__name__)

_emotion_analyzer = EmotionLexiconAnalyzer()


# ---------------------------------------------------------
# Utility Helpers
# ---------------------------------------------------------

def _safe_token(value: object) -> str:
    """
    Normalize tokens for safe feature insertion.
    """

    token = str(value).strip().lower()

    token = token.replace(" ", "_").replace(".", "_").replace("-", "_")

    return "".join(ch for ch in token if ch.isalnum() or ch == "_")


# ---------------------------------------------------------
# Metadata Token Generator
# ---------------------------------------------------------

def _metadata_token_block(row: pd.Series) -> str:
    """
    Convert metadata features into token strings.
    """

    word_count = int(row.get("word_count", 0))
    sentence_count = int(row.get("sentence_count", 0))
    exclamation_count = int(row.get("exclamation_count", 0))
    question_count = int(row.get("question_count", 0))
    uppercase_ratio = float(row.get("uppercase_ratio", 0.0))
    source_credibility = int(row.get("source_credibility", 0))
    is_high_credibility = int(row.get("is_high_credibility", 0))
    is_low_credibility = int(row.get("is_low_credibility", 0))

    domain = row.get("domain", "unknown")

    domain_token = _safe_token(domain) or "unknown"

    return " ".join(
        [
            f"meta_wcbin_{min(word_count // 100, 50)}",
            f"meta_scbin_{min(sentence_count // 10, 50)}",
            f"meta_excbin_{min(exclamation_count, 10)}",
            f"meta_qbin_{min(question_count, 10)}",
            f"meta_upper_{min(int(uppercase_ratio * 10), 10)}",
            f"meta_srccred_{source_credibility}",
            f"meta_is_high_{is_high_credibility}",
            f"meta_is_low_{is_low_credibility}",
            f"meta_domain_{domain_token}",
        ]
    )


# ---------------------------------------------------------
# TF-IDF Keyword Extraction
# ---------------------------------------------------------

def _top_tfidf_terms_for_row(
    matrix_row: Any,
    feature_names: list[str],
    top_terms_per_doc: int,
) -> str:
    """
    Extract top TF-IDF keywords for a document.
    """

    if matrix_row.nnz == 0:
        return ""

    sorted_indices = np.argsort(matrix_row.data)[::-1][:top_terms_per_doc]

    feature_indices = matrix_row.indices[sorted_indices]

    tokens = [f"kw_{_safe_token(feature_names[idx])}" for idx in feature_indices]

    return " ".join(tokens)


# ---------------------------------------------------------
# Bias + Emotion Tokens
# ---------------------------------------------------------

def _clip_bin(value: float, factor: float = 10.0, max_bin: int = 20) -> int:
    return max(0, min(int(value * factor), max_bin))


def _bias_emotion_token_block(text: str) -> str:
    """
    Convert bias and emotion signals into token features.
    """

    text = str(text or "")

    bias_result = compute_bias_features(text)

    emotion_result = _emotion_analyzer.analyze(text)

    dominant_emotion = _safe_token(emotion_result.dominant_emotion or "neutral")

    media_bias = _safe_token(bias_result.media_bias or "neutral")

    sorted_emotions = sorted(
        emotion_result.emotion_scores.items(),
        key=lambda item: item[1],
        reverse=True,
    )

    top_emotions = [name for name, score in sorted_emotions if score > 0][:2]

    tokens = [
        f"sem_bias_{_clip_bin(bias_result.bias_score)}",
        f"sem_media_{media_bias}",
        f"sem_emotion_{dominant_emotion}",
    ]

    for token in bias_result.biased_tokens[:3]:
        tokens.append(f"sem_bias_tok_{_safe_token(token)}")

    for emotion_name in top_emotions:
        tokens.append(f"sem_top_emo_{_safe_token(emotion_name)}")

    return " ".join(tokens)


# ---------------------------------------------------------
# Training Pipeline
# ---------------------------------------------------------

def fit_feature_pipeline(
    train_df: pd.DataFrame,
    text_column: str = "text",
    tfidf_max_features: int = 5000,
    top_terms_per_doc: int = 4,
) -> Tuple[pd.DataFrame, TfidfVectorizer]:
    """
    Fit feature pipeline on training dataset only.
    """

    ensure_dataframe(train_df, name="train_df", required_columns=[text_column])

    ensure_non_empty_text_column(train_df, text_column, name="train_df")

    ensure_positive_int(tfidf_max_features, name="tfidf_max_features", min_value=10)

    ensure_positive_int(top_terms_per_doc, name="top_terms_per_doc", min_value=1)

    logger.info("Fitting feature pipeline on training split")

    featured_df = add_source_features(train_df.copy())

    featured_df = extract_metadata_features(featured_df)

    vectorizer = build_tfidf_vectorizer(max_features=tfidf_max_features)

    tfidf_matrix, vectorizer = tfidf_fit_transform(
        featured_df[text_column].astype(str).tolist(),
        vectorizer=vectorizer,
    )

    feature_names = get_feature_names(vectorizer)

    keyword_tokens = [
        _top_tfidf_terms_for_row(
            tfidf_matrix.getrow(i),
            feature_names,
            top_terms_per_doc,
        )
        for i in range(tfidf_matrix.shape[0])
    ]

    metadata_tokens = featured_df.apply(_metadata_token_block, axis=1)

    semantic_tokens = featured_df[text_column].astype(str).apply(_bias_emotion_token_block)

    featured_df["feature_tokens"] = (
        metadata_tokens
        + " "
        + pd.Series(keyword_tokens, index=featured_df.index)
        + " "
        + semantic_tokens
    ).str.strip()

    featured_df["engineered_text"] = (
        featured_df[text_column].astype(str)
        + " [FEATURES] "
        + featured_df["feature_tokens"]
    )

    logger.info(
        "Feature pipeline completed | rows=%s | tfidf_terms=%s",
        len(featured_df),
        len(feature_names),
    )

    return featured_df, vectorizer


# ---------------------------------------------------------
# Transform Pipeline
# ---------------------------------------------------------

def transform_feature_pipeline(
    df: pd.DataFrame,
    vectorizer: TfidfVectorizer,
    text_column: str = "text",
    top_terms_per_doc: int = 4,
) -> pd.DataFrame:
    """
    Apply trained feature pipeline to new dataset.
    """

    ensure_dataframe(df, name="df", required_columns=[text_column])

    ensure_non_empty_text_column(df, text_column, name="df")

    logger.info("Applying trained feature pipeline")

    featured_df = add_source_features(df.copy())

    featured_df = extract_metadata_features(featured_df)

    tfidf_matrix = tfidf_transform(
        featured_df[text_column].astype(str).tolist(),
        vectorizer=vectorizer,
    )

    feature_names = get_feature_names(vectorizer)

    keyword_tokens = [
        _top_tfidf_terms_for_row(
            tfidf_matrix.getrow(i),
            feature_names,
            top_terms_per_doc,
        )
        for i in range(tfidf_matrix.shape[0])
    ]

    metadata_tokens = featured_df.apply(_metadata_token_block, axis=1)

    semantic_tokens = featured_df[text_column].astype(str).apply(_bias_emotion_token_block)

    featured_df["feature_tokens"] = (
        metadata_tokens
        + " "
        + pd.Series(keyword_tokens, index=featured_df.index)
        + " "
        + semantic_tokens
    ).str.strip()

    featured_df["engineered_text"] = (
        featured_df[text_column].astype(str)
        + " [FEATURES] "
        + featured_df["feature_tokens"]
    )

    logger.info("Feature transform completed | rows=%s", len(featureed_df))

    return featured_df


# ---------------------------------------------------------
# Save Vectorizer
# ---------------------------------------------------------

def save_vectorizer(vectorizer: TfidfVectorizer, output_path: str | Path) -> Path:
    """
    Persist TF-IDF vectorizer to disk.
    """

    output_path = Path(output_path)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    joblib.dump(vectorizer, output_path)

    logger.info("Saved TF-IDF vectorizer to %s", output_path)

    return output_path