"""
Utilities for validating and normalizing the unified multi-task dataset schema.

Canonical 7-task structure:
- bias detection: bias_label
- ideology detection: ideology_label
- propaganda detection: propaganda_label
- frame classification: frame
- narrative role extraction: hero, villain, victim
- narrative frame detection: CO, EC, HI, MO, RE
- emotion detection: emotion_0 ... emotion_19
"""

from __future__ import annotations

from typing import Any, Callable

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
    "neutral": 1,
    "right": 2,
}

PROPAGANDA_LABEL_TO_ID = {
    "no": 0,
    "false": 0,
    "yes": 1,
    "true": 1,
}

TEXT_COLUMNS = ("title", "text")
CLASSIFICATION_COLUMNS = ("bias_label", "ideology_label", "propaganda_label")
FRAME_COLUMN = "frame"
NARRATIVE_FRAME_COLUMNS = ("CO", "EC", "HI", "MO", "RE")
NARRATIVE_ROLE_COLUMNS = ("hero", "villain", "victim")
NARRATIVE_ENTITY_COLUMNS = ("hero_entities", "villain_entities", "victim_entities")
EMOTION_COLUMNS = tuple(f"emotion_{idx}" for idx in range(20))
METADATA_COLUMNS = ("dataset",)

UNIFIED_REQUIRED_COLUMNS = (
    *TEXT_COLUMNS,
    *CLASSIFICATION_COLUMNS,
    FRAME_COLUMN,
    *NARRATIVE_FRAME_COLUMNS,
    *NARRATIVE_ROLE_COLUMNS,
    *NARRATIVE_ENTITY_COLUMNS,
    *EMOTION_COLUMNS,
    *METADATA_COLUMNS,
)

TASK_COLUMN_GROUPS = {
    "text_input": TEXT_COLUMNS,
    "bias_detection": ("bias_label",),
    "ideology_detection": ("ideology_label",),
    "propaganda_detection": ("propaganda_label",),
    "frame_classification": (FRAME_COLUMN,),
    "narrative_frame_detection": NARRATIVE_FRAME_COLUMNS,
    "narrative_role_extraction": (*NARRATIVE_ROLE_COLUMNS, *NARRATIVE_ENTITY_COLUMNS),
    "emotion_detection": EMOTION_COLUMNS,
    "metadata": METADATA_COLUMNS,
}

_COLUMN_ALIASES = {
    "bias": "bias_label",
    "ideology": "ideology_label",
    "propaganda": "propaganda_label",
    "narrative_hero": "hero",
    "narrative_villain": "villain",
    "narrative_victim": "victim",
    "dataset_source": "dataset",
    "co": "CO",
    "ec": "EC",
    "hi": "HI",
    "mo": "MO",
    "re": "RE",
}


def _is_missing(value: Any) -> bool:
    return bool(pd.isna(value))


def _coerce_integer(value: Any) -> int | None:
    if _is_missing(value):
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


def _coerce_optional_categorical(
    value: Any,
    *,
    text_map: dict[str, int],
    allowed_ids: set[int],
) -> int | None:
    if _is_missing(value):
        return None

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


def _coerce_optional_binary_flag(value: Any) -> int | None:
    if _is_missing(value):
        return None

    as_int = _coerce_integer(value)
    if as_int is not None:
        return as_int if as_int in {0, 1} else None

    text = str(value).strip().lower()
    if text in {"yes", "true", "y"}:
        return 1
    if text in {"no", "false", "n"}:
        return 0

    return None


def _coerce_optional_frame(value: Any) -> int | str | None:
    if _is_missing(value):
        return None

    as_int = _coerce_integer(value)
    if as_int is not None:
        return as_int

    text = str(value).strip()
    if not text:
        return None
    return text


def _coerce_emotion_id(value: Any) -> int | None:
    as_int = _coerce_integer(value)
    if as_int is None:
        return None
    if as_int < 0 or as_int > 19:
        return None
    return as_int


def _apply_aliases(df: pd.DataFrame) -> pd.DataFrame:
    normalized = df.copy()

    for alias, canonical in _COLUMN_ALIASES.items():
        if alias in normalized.columns and canonical not in normalized.columns:
            normalized[canonical] = normalized[alias]

    return normalized


def _ensure_required_columns(df: pd.DataFrame) -> pd.DataFrame:
    normalized = df.copy()

    if "text" not in normalized.columns:
        raise ValueError("Missing required unified dataset column: 'text'")

    if "title" not in normalized.columns:
        normalized["title"] = ""

    if "dataset" not in normalized.columns:
        normalized["dataset"] = "unknown"

    for column in UNIFIED_REQUIRED_COLUMNS:
        if column not in normalized.columns:
            normalized[column] = pd.NA

    return normalized


def _expand_legacy_emotion_column(df: pd.DataFrame, errors: list[str]) -> pd.DataFrame:
    normalized = df.copy()

    if "emotion" not in normalized.columns:
        return normalized

    has_any_emotion_column = any(col in normalized.columns for col in EMOTION_COLUMNS)
    if has_any_emotion_column:
        return normalized

    for emotion_col in EMOTION_COLUMNS:
        normalized[emotion_col] = pd.NA

    invalid_rows: list[int] = []
    for idx, value in normalized["emotion"].items():
        if _is_missing(value):
            continue

        emotion_id = _coerce_emotion_id(value)
        if emotion_id is None:
            invalid_rows.append(int(idx))
            continue

        for emotion_col in EMOTION_COLUMNS:
            normalized.at[idx, emotion_col] = 0
        normalized.at[idx, f"emotion_{emotion_id}"] = 1

    if invalid_rows:
        preview = invalid_rows[:5]
        errors.append(
            "emotion: invalid class IDs at rows "
            f"{preview}{'...' if len(invalid_rows) > len(preview) else ''}"
        )

    return normalized


def _apply_optional_numeric_column(
    df: pd.DataFrame,
    *,
    column: str,
    mapper: Callable[[Any], int | None],
    errors: list[str],
) -> pd.DataFrame:
    normalized = df.copy()
    mapped_values: list[int | pd.NA] = []
    invalid_rows: list[int] = []

    for idx, value in normalized[column].items():
        if _is_missing(value):
            mapped_values.append(pd.NA)
            continue

        mapped = mapper(value)
        if mapped is None:
            mapped_values.append(pd.NA)
            invalid_rows.append(int(idx))
            continue

        mapped_values.append(mapped)

    normalized[column] = pd.Series(mapped_values, index=normalized.index, dtype="Int64")

    if invalid_rows:
        preview = invalid_rows[:5]
        errors.append(
            f"{column}: invalid values at rows {preview}"
            + ("..." if len(invalid_rows) > len(preview) else "")
        )

    return normalized


def _normalize_text_and_metadata(df: pd.DataFrame, errors: list[str]) -> pd.DataFrame:
    normalized = df.copy()

    normalized["title"] = normalized["title"].fillna("").astype(str).str.strip()
    normalized["text"] = normalized["text"].fillna("").astype(str).str.strip()
    normalized["dataset"] = normalized["dataset"].fillna("unknown").astype(str).str.strip()

    empty_text_rows = normalized.index[normalized["text"] == ""].tolist()
    if empty_text_rows:
        preview = empty_text_rows[:5]
        errors.append(
            f"text: empty values at rows {preview}"
            + ("..." if len(empty_text_rows) > len(preview) else "")
        )

    empty_dataset_rows = normalized.index[normalized["dataset"] == ""].tolist()
    if empty_dataset_rows:
        preview = empty_dataset_rows[:5]
        errors.append(
            f"dataset: empty values at rows {preview}"
            + ("..." if len(empty_dataset_rows) > len(preview) else "")
        )

    return normalized


def _normalize_entity_columns(df: pd.DataFrame) -> pd.DataFrame:
    normalized = df.copy()
    for column in NARRATIVE_ENTITY_COLUMNS:
        normalized[column] = normalized[column].apply(
            lambda value: pd.NA if _is_missing(value) else str(value).strip()
        )
    return normalized


def _normalize_frame_column(df: pd.DataFrame, errors: list[str]) -> pd.DataFrame:
    normalized = df.copy()

    mapped_values: list[int | str | pd.NA] = []
    invalid_rows: list[int] = []
    for idx, value in normalized[FRAME_COLUMN].items():
        if _is_missing(value):
            mapped_values.append(pd.NA)
            continue

        mapped = _coerce_optional_frame(value)
        if mapped is None:
            mapped_values.append(pd.NA)
            invalid_rows.append(int(idx))
            continue

        mapped_values.append(mapped)

    normalized[FRAME_COLUMN] = pd.Series(mapped_values, index=normalized.index, dtype="object")

    if invalid_rows:
        preview = invalid_rows[:5]
        errors.append(
            f"{FRAME_COLUMN}: invalid values at rows {preview}"
            + ("..." if len(invalid_rows) > len(preview) else "")
        )

    return normalized


def normalize_unified_labels(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize and validate labels for the canonical 7-task unified dataset.
    """

    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame")

    errors: list[str] = []

    normalized = _apply_aliases(df)
    normalized = _expand_legacy_emotion_column(normalized, errors)
    normalized = _ensure_required_columns(normalized)
    normalized = _normalize_text_and_metadata(normalized, errors)

    normalized = _apply_optional_numeric_column(
        normalized,
        column="bias_label",
        mapper=lambda value: _coerce_optional_categorical(
            value,
            text_map=BIAS_LABEL_TO_ID,
            allowed_ids={0, 1},
        ),
        errors=errors,
    )
    normalized = _apply_optional_numeric_column(
        normalized,
        column="ideology_label",
        mapper=lambda value: _coerce_optional_categorical(
            value,
            text_map=IDEOLOGY_LABEL_TO_ID,
            allowed_ids={0, 1, 2},
        ),
        errors=errors,
    )
    normalized = _apply_optional_numeric_column(
        normalized,
        column="propaganda_label",
        mapper=lambda value: _coerce_optional_categorical(
            value,
            text_map=PROPAGANDA_LABEL_TO_ID,
            allowed_ids={0, 1},
        ),
        errors=errors,
    )

    normalized = _normalize_frame_column(normalized, errors)
    normalized = _normalize_entity_columns(normalized)

    for column in (*NARRATIVE_FRAME_COLUMNS, *NARRATIVE_ROLE_COLUMNS, *EMOTION_COLUMNS):
        normalized = _apply_optional_numeric_column(
            normalized,
            column=column,
            mapper=_coerce_optional_binary_flag,
            errors=errors,
        )

    if errors:
        raise ValueError("Unified label normalization failed: " + "; ".join(errors))

    canonical_prefix = list(UNIFIED_REQUIRED_COLUMNS)
    extra_columns = [col for col in normalized.columns if col not in canonical_prefix]
    normalized = normalized[canonical_prefix + extra_columns]

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
