"""
File: src/data/class_balance.py

Purpose
-------
Provide utilities for detecting and correcting dataset class imbalance.

Supports:
- Class distribution inspection
- Random oversampling
- Random undersampling
- Automatic dataset balancing

Used in fake news detection pipelines where label imbalance
can significantly degrade model performance.

Inputs
------
df : pandas.DataFrame
label_column : str

Outputs
-------
Balanced pandas.DataFrame

Dependencies
------------
pandas
sklearn
logging
"""

from __future__ import annotations

import logging
from typing import Dict

import pandas as pd
from sklearn.utils import resample

logger = logging.getLogger(__name__)


# -------------------------------------------------
# Class Distribution Check
# -------------------------------------------------

def check_class_distribution(
    df: pd.DataFrame,
    label_column: str = "label",
) -> Dict[int, int]:
    """
    Inspect class distribution.

    Parameters
    ----------
    df : pandas.DataFrame
    label_column : str

    Returns
    -------
    Dict[int, int]
        Label counts.
    """

    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame")

    if label_column not in df.columns:
        raise ValueError(
            f"Column '{label_column}' not found in dataset"
        )

    if len(df) == 0:
        raise ValueError("Dataset is empty")

    counts = df[label_column].value_counts().to_dict()

    logger.info("Class distribution: %s", counts)

    return counts


# -------------------------------------------------
# Random Oversampling
# -------------------------------------------------

def random_oversample(
    df: pd.DataFrame,
    label_column: str = "label",
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Balance dataset using random oversampling.

    Minority class samples are duplicated
    until all classes match the majority count.
    """

    if len(df) == 0:
        raise ValueError("Dataset is empty")

    counts = df[label_column].value_counts()
    if len(counts) < 2:
        logger.warning("Only one class present; skipping oversampling")
        return df.reset_index(drop=True)

    max_count = counts.max()

    balanced_frames = []

    for label in counts.index:

        class_df = df[df[label_column] == label]

        resampled = resample(
            class_df,
            replace=True,
            n_samples=max_count,
            random_state=random_state,
        )

        balanced_frames.append(resampled)

    balanced_df = pd.concat(balanced_frames)

    balanced_df = balanced_df.sample(
        frac=1,
        random_state=random_state,
    ).reset_index(drop=True)

    logger.info(
        "Dataset balanced via oversampling: %s",
        balanced_df[label_column].value_counts().to_dict(),
    )

    return balanced_df


# -------------------------------------------------
# Random Undersampling
# -------------------------------------------------

def random_undersample(
    df: pd.DataFrame,
    label_column: str = "label",
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Balance dataset using random undersampling.

    Majority classes are reduced to match
    the minority class size.
    """

    if len(df) == 0:
        raise ValueError("Dataset is empty")

    counts = df[label_column].value_counts()
    if len(counts) < 2:
        logger.warning("Only one class present; skipping undersampling")
        return df.reset_index(drop=True)

    min_count = counts.min()

    balanced_frames = []

    for label in counts.index:

        class_df = df[df[label_column] == label]

        resampled = resample(
            class_df,
            replace=False,
            n_samples=min_count,
            random_state=random_state,
        )

        balanced_frames.append(resampled)

    balanced_df = pd.concat(balanced_frames)

    balanced_df = balanced_df.sample(
        frac=1,
        random_state=random_state,
    ).reset_index(drop=True)

    logger.info(
        "Dataset balanced via undersampling: %s",
        balanced_df[label_column].value_counts().to_dict(),
    )

    return balanced_df


# -------------------------------------------------
# Automatic Balancing
# -------------------------------------------------

def balance_dataset(
    df: pd.DataFrame,
    label_column: str = "label",
    method: str = "oversample",
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Automatically balance dataset.

    Parameters
    ----------
    df : pandas.DataFrame

    label_column : str

    method : str
        "oversample" or "undersample"

    Returns
    -------
    pandas.DataFrame
        Balanced dataset
    """

    if method == "oversample":

        return random_oversample(df, label_column, random_state=random_state)

    elif method == "undersample":

        return random_undersample(df, label_column, random_state=random_state)

    else:

        raise ValueError(
            "method must be 'oversample' or 'undersample'"
        )
