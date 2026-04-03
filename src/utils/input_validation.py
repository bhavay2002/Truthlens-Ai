"""
File Name: input_validation.py
Module: src.utils
Description:
    Input validation utilities for TruthLens AI.

    This module provides reusable validation functions to ensure
    data integrity across the ML pipeline. It includes validation
    for pandas DataFrames, scalar parameters, and text inputs used
    throughout training, inference, and preprocessing pipelines.

Author: TruthLens Engineering
Date: 2026-04-03
Dependencies:
    - Python 3.10+
    - pandas

Inputs:
    - DataFrames
    - text inputs
    - numeric parameters

Outputs:
    - validated values
    - raised exceptions for invalid inputs
"""

from __future__ import annotations

import logging
from typing import Iterable, Sequence, Any

import pandas as pd


# ---------------------------------------------------------
# Logging
# ---------------------------------------------------------

logger = logging.getLogger(__name__)


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
        Input DataFrame to validate.

    name : str
        Variable name used in error messages.

    required_columns : Iterable[str]
        Columns that must exist in the DataFrame.

    min_rows : int
        Minimum number of rows required.

    Raises
    ------
    TypeError
        If input is not a DataFrame.

    ValueError
        If DataFrame is empty or missing required columns.
    """

    if not isinstance(df, pd.DataFrame):
        logger.error("%s must be a pandas DataFrame", name)
        raise TypeError(f"{name} must be a pandas DataFrame")

    if len(df) < min_rows:
        logger.error("%s contains fewer than %d rows", name, min_rows)
        raise ValueError(f"{name} must contain at least {min_rows} row(s)")

    missing_columns = set(required_columns) - set(df.columns)

    if missing_columns:
        logger.error("%s missing required columns: %s", name, missing_columns)
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

    Parameters
    ----------
    value : int
        Value to validate.

    name : str
        Parameter name.

    min_value : int
        Minimum allowed value.

    Returns
    -------
    int
        Validated integer.

    Raises
    ------
    TypeError
        If value is not an integer.

    ValueError
        If value is below minimum.
    """

    if isinstance(value, bool) or not isinstance(value, int):
        logger.error("%s must be an integer", name)
        raise TypeError(f"{name} must be an integer")

    if value < min_value:
        logger.error("%s must be >= %d", name, min_value)
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

    Parameters
    ----------
    df : pd.DataFrame
        Dataset.

    text_column : str
        Column containing text.

    name : str
        Variable name for error reporting.

    Raises
    ------
    ValueError
        If column missing or contains only empty values.
    """

    if text_column not in df.columns:
        logger.error("%s does not contain column '%s'", name, text_column)
        raise ValueError(
            f"{name} does not contain text column '{text_column}'"
        )

    def _normalize(value: Any) -> str:
        if value is None:
            return ""

        try:
            if bool(pd.isna(value)):
                return ""
        except Exception:
            pass

        return str(value).strip()

    if df[text_column].map(_normalize).eq("").all():
        logger.error("%s.%s contains only empty values", name, text_column)
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

    Parameters
    ----------
    text : str
        Input text.

    name : str
        Variable name.

    Returns
    -------
    str
        Validated text.

    Raises
    ------
    TypeError
        If input is not a string.

    ValueError
        If text is empty.
    """

    if not isinstance(text, str):
        logger.error("%s must be a string", name)
        raise TypeError(f"{name} must be a string")

    if not text.strip():
        logger.error("%s cannot be empty", name)
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

    Parameters
    ----------
    texts : Sequence[str] | Iterable[str]
        Iterable of text values.

    name : str
        Variable name.

    Returns
    -------
    list[str]
        Normalized list of text values.

    Raises
    ------
    ValueError
        If list is empty or contains only empty strings.
    """

    if texts is None:
        logger.error("%s cannot be None", name)
        raise ValueError(f"{name} cannot be None")

    if isinstance(texts, (str, bytes)):
        iterable: Iterable[Any] = [texts]
    else:
        try:
            iterable = iter(texts)
        except TypeError as exc:
            logger.exception("Invalid iterable for %s", name)
            raise TypeError(
                f"{name} must be an iterable of text values"
            ) from exc

    text_list: list[str] = []

    for item in iterable:
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
        logger.error("%s cannot be empty", name)
        raise ValueError(f"{name} cannot be empty")

    if all(not item.strip() for item in text_list):
        logger.error("%s contains only empty text", name)
        raise ValueError(f"{name} cannot be entirely empty")

    return text_list