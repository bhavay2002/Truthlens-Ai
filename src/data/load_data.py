"""
File: src/data/load_data.py

Purpose
-------
Provide reliable dataset loading utilities for NLP pipelines.

Functions included:
- Load CSV datasets safely
- Merge fake and real news datasets
- Assign labels for binary classification

Label Convention
----------------
1 = Fake News
0 = Real News

Inputs
------
path : str | Path
    Path to CSV dataset

fake_path : str | Path
real_path : str | Path

Outputs
-------
load_csv(path) -> pandas.DataFrame

merge_datasets(fake_path, real_path) -> pandas.DataFrame

Dependencies
------------
pandas
pathlib
logging
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Union

import pandas as pd

logger = logging.getLogger(__name__)


# -------------------------------------------------
# CSV Loader
# -------------------------------------------------

def load_csv(path: Union[str, Path]) -> pd.DataFrame:
    """
    Load a CSV dataset with validation.

    Parameters
    ----------
    path : str | Path
        Path to CSV dataset.

    Returns
    -------
    pd.DataFrame
        Loaded dataset.
    """

    path_obj = Path(path)

    if not path_obj.exists():
        raise FileNotFoundError(f"CSV file not found: {path_obj}")

    try:
        df = pd.read_csv(path_obj)
    except Exception as e:
        logger.error("Failed to load CSV from %s: %s", path_obj, e)
        raise

    logger.info("Loaded %s rows from %s", len(df), path_obj)
    return df


def _prepare_frame(
    df: pd.DataFrame,
    *,
    label_value: int,
    text_column: str,
    title_column: str,
    label_column: str,
) -> pd.DataFrame:
    if text_column not in df.columns:
        raise ValueError(f"Missing required text column '{text_column}'")

    prepared = df.copy()

    if title_column not in prepared.columns:
        prepared[title_column] = ""

    prepared[text_column] = prepared[text_column].astype(str).fillna("").str.strip()
    prepared[title_column] = prepared[title_column].astype(str).fillna("").str.strip()
    prepared[label_column] = int(label_value)

    prepared = prepared[prepared[text_column].str.len() > 0]

    return prepared[[title_column, text_column, label_column]]


# -------------------------------------------------
# Dataset Merger
# -------------------------------------------------

def merge_datasets(
    fake_path: Union[str, Path],
    real_path: Union[str, Path],
    *,
    text_column: str = "text",
    title_column: str = "title",
    label_column: str = "label",
    fake_label: int = 1,
    real_label: int = 0,
) -> pd.DataFrame:
    """
    Merge fake news and real news datasets.

    Automatically assigns labels:
        Fake = 1
        Real = 0

    Parameters
    ----------
    fake_path : str | Path
        Path to fake news dataset.

    real_path : str | Path
        Path to real news dataset.

    Returns
    -------
    pd.DataFrame
        Combined labeled dataset.
    """

    fake_df = load_csv(fake_path)
    real_df = load_csv(real_path)

    logger.info("Loaded %s fake news articles", len(fake_df))
    logger.info("Loaded %s real news articles", len(real_df))

    fake_prepared = _prepare_frame(
        fake_df,
        label_value=fake_label,
        text_column=text_column,
        title_column=title_column,
        label_column=label_column,
    )
    real_prepared = _prepare_frame(
        real_df,
        label_value=real_label,
        text_column=text_column,
        title_column=title_column,
        label_column=label_column,
    )

    merged_df = pd.concat([fake_prepared, real_prepared], ignore_index=True)

    logger.info(
        "Merged dataset contains %s total articles after cleanup",
        len(merged_df),
    )

    return merged_df
