"""
Utilities for validating and normalizing unified multi-task labels.

Supported schema:
- bias: 0/1 (Non-biased/Biased)
- ideology: 0/1/2 (Left/Center/Right)
- propaganda: 0/1 (No/Yes)
- narrative flags: narrative_hero, narrative_villain, narrative_victim (0/1)
- emotion: integer in [0, 19]
"""

from __future__ import annotations

import re
from typing import Any

import pandas as pd


BIAS_LABEL_TO_ID = {
    "non-biased": 0,
    "non biased": 0,
    "nonbiased": 0,
    "neutral": 0,
    "biased": 1,
}

IDEOLOGY_LABEL_TO_ID = {
    "left": 0,
    "center": 1,
    "centre": 1,
    "right": 2,
}

PROPAGANDA_LABEL_TO_ID = {
    "no": 0,
    "false": 0,
    "yes": 1,
    "true": 1,
}

UNIFIED_REQUIRED_COLUMNS = ("bias", "ideology", "propaganda", "emotion")
UNIFIED_NARRATIVE_FLAG_COLUMNS = (
    "narrative_hero",
    "narrative_villain",
    "narrative_victim",
)

_NARRATIVE_ALIAS_COLUMNS = {
    "hero": "narrative_hero",
    "villain": "narrative_villain",
    "victim": "narrative_victim",
}


def _is_nan(value: Any) -> bool:
    return bool(pd.isna(value))


def _coerce_integer(value: Any) -> int | None:
    if _is_nan(value):
        return None

    if isinstance(value, bool):
        return int(value)

    if isinstance(value, int):
        return value

    if isinstance(value, float):
        if not value.is_integer():
            return None
        return int(value)

    text = str(value).strip()
    if not text:
        return None

    if text.lstrip("-").isdigit():
        return int(text)

    return None


def _coerce_categorical(
    value: Any,
    *,
    text_map: dict[str, int],
    allowed_ids: set[int],
) -> int | None:
    as_int = _coerce_integer(value)
    if as_int is not None:
        return as_int if as_int in allowed_ids else None

    text = str(value).strip().lower()
    if not text:
        return None

    mapped = text_map.get(text)
    if mapped is None:
        return None

    return mapped


def _coerce_binary_flag(value: Any) -> int | None:
    as_int = _coerce_integer(value)
    if as_int is not None:
        return as_int if as_int in {0, 1} else None

    text = str(value).strip().lower()
    if text in {"yes", "true", "y"}:
        return 1
    if text in {"no", "false", "n"}:
        return 0

    return None


def _coerce_emotion_id(value: Any) -> int | None:
    as_int = _coerce_integer(value)
    if as_int is None:
        return None
    if as_int < 0 or as_int > 19:
        return None
    return as_int


def _expand_narrative_column(df: pd.DataFrame) -> pd.DataFrame:
    if "narrative" not in df.columns:
        return df

    updated = df.copy()

    for target in UNIFIED_NARRATIVE_FLAG_COLUMNS:
        if target not in updated.columns:
            updated[target] = 0

    for idx, value in updated["narrative"].items():
        if _is_nan(value):
            continue

        tokens = {
            token.strip().lower()
            for token in re.split(r"[,|;/]", str(value))
            if token.strip()
        }
        if "hero" in tokens:
            updated.at[idx, "narrative_hero"] = 1
        if "villain" in tokens:
            updated.at[idx, "narrative_villain"] = 1
        if "victim" in tokens:
            updated.at[idx, "narrative_victim"] = 1

    return updated


def normalize_unified_labels(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize unified labels into numeric IDs and narrative binary flags.
    """

    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame")

    normalized = df.copy()
    normalized = _expand_narrative_column(normalized)

    for alias, canonical in _NARRATIVE_ALIAS_COLUMNS.items():
        if alias in normalized.columns and canonical not in normalized.columns:
            normalized[canonical] = normalized[alias]

    errors: list[str] = []

    missing_required = [
        col for col in UNIFIED_REQUIRED_COLUMNS if col not in normalized.columns
    ]
    if missing_required:
        raise ValueError(
            "Missing required unified label columns: "
            f"{missing_required}."
        )

    for col in UNIFIED_NARRATIVE_FLAG_COLUMNS:
        if col not in normalized.columns:
            normalized[col] = 0

    def _apply(
        column: str,
        mapper,
    ) -> None:
        mapped_values: list[int | None] = []
        invalid_rows: list[int] = []

        for idx, value in normalized[column].items():
            mapped = mapper(value)
            mapped_values.append(mapped)
            if mapped is None:
                invalid_rows.append(int(idx))

        normalized[column] = mapped_values

        if invalid_rows:
            preview = invalid_rows[:5]
            errors.append(
                f"{column}: invalid values at rows {preview}"
                + ("..." if len(invalid_rows) > len(preview) else "")
            )

    _apply(
        "bias",
        lambda value: _coerce_categorical(
            value,
            text_map=BIAS_LABEL_TO_ID,
            allowed_ids={0, 1},
        ),
    )
    _apply(
        "ideology",
        lambda value: _coerce_categorical(
            value,
            text_map=IDEOLOGY_LABEL_TO_ID,
            allowed_ids={0, 1, 2},
        ),
    )
    _apply(
        "propaganda",
        lambda value: _coerce_categorical(
            value,
            text_map=PROPAGANDA_LABEL_TO_ID,
            allowed_ids={0, 1},
        ),
    )
    _apply(
        "emotion",
        _coerce_emotion_id,
    )

    for col in UNIFIED_NARRATIVE_FLAG_COLUMNS:
        _apply(col, _coerce_binary_flag)

    if errors:
        raise ValueError(
            "Unified label normalization failed: " + "; ".join(errors)
        )

    for col in (
        *UNIFIED_REQUIRED_COLUMNS,
        *UNIFIED_NARRATIVE_FLAG_COLUMNS,
    ):
        normalized[col] = normalized[col].astype(int)

    return normalized


def validate_unified_labels(df: pd.DataFrame) -> list[str]:
    """
    Return validation errors for unified labels.
    Empty list means the schema is valid.
    """

    try:
        normalize_unified_labels(df)
        return []
    except ValueError as exc:
        return [str(exc)]
