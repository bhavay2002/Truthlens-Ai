"""
File: input_validation.py

Purpose
-------
Input validation utilities for TruthLens AI.

This module provides reusable validation functions for ensuring
data integrity across the ML pipeline.
"""

from __future__ import annotations

from typing import Iterable, Sequence

import pandas as pd


# ---------------------------------------------------------
# DataFrame Validation
# ---------------------------------------------------------

def ensure_dataframe(
    df: pd.DataFrame,
    *,
    name: str = "df",
    required_columns: Iterable[str] = (),
    min_rows: int = 1,
) -> None:
    """
    Validate pandas DataFrame input.

    Parameters
    ----------
    df : pd.DataFrame

    required_columns : Iterable[str]

    min_rows : int
    """

    if not isinstance(df, pd.DataFrame):

        raise TypeError(f"{name} must be a pandas DataFrame")

    if len(df) < min_rows:

        raise ValueError(f"{name} must contain at least {min_rows} row(s)")

    missing_columns = set(required_columns) - set(df.columns)

    if missing_columns:

        raise ValueError(
            f"{name} is missing required columns: {sorted(missing_columns)}"
        )


# ---------------------------------------------------------
# Positive Integer Validation
# ---------------------------------------------------------

def ensure_positive_int(
    value: int,
    *,
    name: str,
    min_value: int = 1,
) -> int:
    """
    Ensure integer parameter is valid.
    """

    if not isinstance(value, int):

        raise TypeError(f"{name} must be an integer")

    if value < min_value:

        raise ValueError(f"{name} must be >= {min_value}")

    return value


# ---------------------------------------------------------
# Text Column Validation
# ---------------------------------------------------------

def ensure_non_empty_text_column(
    df: pd.DataFrame,
    text_column: str,
    *,
    name: str = "df",
) -> None:
    """
    Ensure dataset text column exists and contains valid text.
    """

    if text_column not in df.columns:

        raise ValueError(
            f"{name} does not contain text column "
            f"'{text_column}'"
        )

    def _normalize(value: object) -> str:
        if value is None:
            return ""

        try:
            if bool(pd.isna(value)):
                return ""
        except Exception:
            pass

        return str(value).strip()

    if df[text_column].map(_normalize).eq("").all():

        raise ValueError(f"{name}.{text_column} cannot be entirely empty")


# ---------------------------------------------------------
# Single Text Validation
# ---------------------------------------------------------

def ensure_non_empty_text(
    text: str,
    *,
    name: str = "text",
) -> str:
    """
    Validate single text input.
    """

    if not isinstance(text, str):

        raise TypeError(f"{name} must be a string")

    if not text.strip():

        raise ValueError(f"{name} cannot be empty")

    return text


# ---------------------------------------------------------
# Text List Validation
# ---------------------------------------------------------

def ensure_non_empty_text_list(
    texts: Sequence[str] | Iterable[str],
    *,
    name: str = "texts",
) -> list[str]:
    """
    Validate list of text inputs.
    """

    text_list: list[str] = []

    for item in texts:
        if item is None:
            text_list.append("")
            continue

        try:
            if bool(pd.isna(item)):
                text_list.append("")
                continue
        except Exception:
            pass

        text_list.append(str(item))

    if not text_list:

        raise ValueError(f"{name} cannot be empty")

    if all(not item.strip() for item in text_list):

        raise ValueError(f"{name} cannot be entirely empty")

    return text_list
