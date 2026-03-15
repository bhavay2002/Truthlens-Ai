from __future__ import annotations

import pandas as pd
import pytest

from src.models import train_roberta


def _training_df(rows: int = 100) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "id": list(range(rows)),
            "text": [f"sample training text {i}" for i in range(rows)],
            "label": [i % 2 for i in range(rows)],
        }
    )


def test_split_train_val_test_produces_disjoint_partitions() -> None:
    df = _training_df(100)

    train_df, val_df, test_df = train_roberta._split_train_val_test(df)

    assert len(train_df) + len(val_df) + len(test_df) == len(df)
    assert set(train_df["id"]).isdisjoint(val_df["id"])
    assert set(train_df["id"]).isdisjoint(test_df["id"])
    assert set(val_df["id"]).isdisjoint(test_df["id"])
    assert set(train_df.columns) >= {"text", "label"}


def test_validate_split_df_requires_expected_columns() -> None:
    missing_text_df = pd.DataFrame({"label": [0, 1]})

    with pytest.raises(ValueError):
        train_roberta._validate_split_df(missing_text_df, "df", "text")


def test_validate_split_df_rejects_empty_text_column() -> None:
    empty_text_df = pd.DataFrame({"text": ["", "   "], "label": [0, 1]})

    with pytest.raises(ValueError):
        train_roberta._validate_split_df(empty_text_df, "df", "text")
