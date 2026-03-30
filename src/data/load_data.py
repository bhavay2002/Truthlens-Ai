"""
File: src/data/load_data.py

Purpose
-------
Reliable dataset loading utilities for NLP pipelines.

Supports:
- safe CSV loading
- schema normalization
- fake/real path-based merge for binary datasets
- dataframe-list merge for multi-source datasets
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Union, List

import pandas as pd

logger = logging.getLogger(__name__)


# -------------------------------------------------
# Safe CSV Loader
# -------------------------------------------------

def load_csv(
    path: Union[str, Path],
    encoding: str = "utf-8",
    low_memory: bool = False,
) -> pd.DataFrame:
    """
    Load CSV safely with logging and validation.
    """

    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"CSV file not found: {path}")

    try:

        df = pd.read_csv(
            path,
            encoding=encoding,
            low_memory=low_memory,
        )

    except Exception as e:

        logger.error("Failed loading CSV %s: %s", path, e)
        raise

    logger.info("Loaded dataset: %s (%d rows)", path.name, len(df))

    return df


# -------------------------------------------------
# Normalize Dataset Schema
# -------------------------------------------------

def normalize_schema(
    df: pd.DataFrame,
    text_column: str = "text",
    title_column: str | None = None,
    label_columns: List[str] | None = None,
) -> pd.DataFrame:
    """
    Normalize dataset schema for downstream pipelines.
    """

    df = df.copy()

    if text_column not in df.columns:
        raise ValueError(f"Missing text column '{text_column}'")

    df[text_column] = df[text_column].fillna("").astype(str)

    if title_column and title_column in df.columns:

        df[text_column] = (
            df[title_column].fillna("").astype(str)
            + " </s> "
            + df[text_column]
        )

    df = df[df[text_column].str.len() > 0]

    if label_columns:

        for col in label_columns:

            if col not in df.columns:

                df[col] = None

    return df


# -------------------------------------------------
# Merge Binary Fake/Real Dataset
# -------------------------------------------------

def merge_fake_real(
    fake_path: Union[str, Path],
    real_path: Union[str, Path],
    text_column: str = "text",
    title_column: str = "title",
    label_column: str = "label",
    fake_label: int = 1,
    real_label: int = 0,
) -> pd.DataFrame:
    """
    Merge fake and real news datasets.

    Fake = fake_label
    Real = real_label
    """

    fake_df = load_csv(fake_path)
    real_df = load_csv(real_path)

    fake_df = normalize_schema(
        fake_df,
        text_column=text_column,
        title_column=title_column,
    )

    real_df = normalize_schema(
        real_df,
        text_column=text_column,
        title_column=title_column,
    )

    fake_df[label_column] = int(fake_label)
    real_df[label_column] = int(real_label)

    merged = pd.concat([fake_df, real_df], ignore_index=True)

    logger.info(
        "Merged dataset created | fake=%d real=%d total=%d",
        len(fake_df),
        len(real_df),
        len(merged),
    )

    return merged


# -------------------------------------------------
# Merge Multiple Datasets
# -------------------------------------------------

def merge_datasets(
    datasets_or_fake_path: List[pd.DataFrame] | str | Path,
    real_path: Union[str, Path, None] = None,
    *,
    text_column: str = "text",
    title_column: str = "title",
    label_column: str = "label",
    fake_label: int = 1,
    real_label: int = 0,
    dataset_names: List[str] | None = None,
) -> pd.DataFrame:
    """
    Backward-compatible merge utility.

    Supports two call styles:
    1) merge_datasets(fake_path, real_path, ...)
    2) merge_datasets([df1, df2, ...], dataset_names=[...])

    For list input, adds optional dataset_source column for traceability.
    """

    if isinstance(datasets_or_fake_path, (str, Path)):
        if real_path is None:
            raise ValueError(
                "real_path is required when merge_datasets is called with fake_path"
            )

        return merge_fake_real(
            fake_path=datasets_or_fake_path,
            real_path=real_path,
            text_column=text_column,
            title_column=title_column,
            label_column=label_column,
            fake_label=fake_label,
            real_label=real_label,
        )

    if not isinstance(datasets_or_fake_path, list):
        raise TypeError(
            "datasets_or_fake_path must be either a list of DataFrames "
            "or a fake dataset path"
        )

    frames = []

    for i, df in enumerate(datasets_or_fake_path):
        if not isinstance(df, pd.DataFrame):
            raise TypeError(f"Expected DataFrame at index {i}, got {type(df).__name__}")

        frame = df.copy()
        if dataset_names:
            frame["dataset_source"] = dataset_names[i]

        frames.append(frame)

    merged = pd.concat(frames, ignore_index=True)

    logger.info(
        "Merged %d datasets | total rows=%d",
        len(frames),
        len(merged),
    )

    return merged
