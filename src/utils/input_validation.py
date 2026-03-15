from __future__ import annotations

from typing import Iterable, Sequence

import pandas as pd


def ensure_dataframe(
    df: pd.DataFrame,
    *,
    name: str = "df",
    required_columns: Iterable[str] = (),
    min_rows: int = 1,
) -> None:
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"{name} must be a pandas DataFrame")

    if len(df) < min_rows:
        raise ValueError(f"{name} must contain at least {min_rows} row(s)")

    missing_columns = set(required_columns) - set(df.columns)
    if missing_columns:
        raise ValueError(f"{name} is missing required columns: {sorted(missing_columns)}")


def ensure_positive_int(value: int, *, name: str, min_value: int = 1) -> int:
    if not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if value < min_value:
        raise ValueError(f"{name} must be >= {min_value}")
    return value


def ensure_non_empty_text_column(df: pd.DataFrame, text_column: str, *, name: str = "df") -> None:
    if text_column not in df.columns:
        raise ValueError(f"{name} does not contain text column '{text_column}'")
    if df[text_column].astype(str).str.strip().eq("").all():
        raise ValueError(f"{name}.{text_column} cannot be entirely empty")


def ensure_non_empty_text(text: str, *, name: str = "text") -> str:
    if not isinstance(text, str):
        raise TypeError(f"{name} must be a string")
    if not text.strip():
        raise ValueError(f"{name} cannot be empty")
    return text


def ensure_non_empty_text_list(
    texts: Sequence[str] | Iterable[str],
    *,
    name: str = "texts",
) -> list[str]:
    text_list = [str(item) for item in texts]
    if not text_list:
        raise ValueError(f"{name} cannot be empty")
    if all(not item.strip() for item in text_list):
        raise ValueError(f"{name} cannot be entirely empty")
    return text_list
