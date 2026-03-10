from __future__ import annotations

import pandas as pd
import pytest

from src.data.data_augmentation import augment_dataset
from src.features.feature_pipeline import apply_feature_engineering
from src.utils.input_validation import ensure_positive_int


def test_augment_dataset_multiplier_one_is_noop():
    df = pd.DataFrame({"text": ["hello world", "another sample"], "label": [0, 1]})
    augmented = augment_dataset(df, text_column="text", multiplier=1)

    assert len(augmented) == len(df)
    assert list(augmented["text"]) == list(df["text"])


def test_augment_dataset_invalid_multiplier_raises():
    df = pd.DataFrame({"text": ["hello world"], "label": [0]})
    with pytest.raises(ValueError):
        augment_dataset(df, text_column="text", multiplier=0)


def test_apply_feature_engineering_adds_engineered_text():
    df = pd.DataFrame(
        {
            "title": ["T1", "T2", "T3", "T4", "T5"],
            "text": [
                "economy inflation jobs report",
                "football match goal championship",
                "space telescope galaxy mission",
                "medical trial vaccine data",
                "finance market stocks bonds",
            ],
            "label": [0, 1, 0, 1, 0],
            "source": [
                "https://bbc.com/a",
                "https://unknown.site/b",
                "https://apnews.com/c",
                "https://reuters.com/d",
                "https://example.com/e",
            ],
        }
    )
    featured_df, vectorizer = apply_feature_engineering(
        df,
        text_column="text",
        tfidf_max_features=50,
        top_terms_per_doc=2,
    )

    assert "engineered_text" in featured_df.columns
    assert featured_df["engineered_text"].str.contains(r"\[FEATURES\]").all()
    assert vectorizer is not None


def test_ensure_positive_int_validation():
    assert ensure_positive_int(2, name="value", min_value=1) == 2
    with pytest.raises(ValueError):
        ensure_positive_int(0, name="value", min_value=1)
