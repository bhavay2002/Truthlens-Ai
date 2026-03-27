"""
File: src/data/class_balance.py

Purpose
-------
Research-grade dataset balancing utilities.

Supports:
- Single-task and multi-task datasets
- Class distribution inspection
- Random oversampling
- Random undersampling
- Automatic balancing
- Multi-label / multi-task balancing
- Missing label handling

Designed for multi-task NLP systems such as:

Bias
Ideology
Propaganda
Narrative
Emotion

Dependencies
------------
pandas
sklearn
logging
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional

import pandas as pd
from sklearn.utils import resample

logger = logging.getLogger(__name__)


# -------------------------------------------------
# Utility Validation
# -------------------------------------------------

def _validate_dataframe(df: pd.DataFrame):

    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame")

    if len(df) == 0:
        raise ValueError("Dataset is empty")


# -------------------------------------------------
# Class Distribution Check
# -------------------------------------------------

def check_class_distribution(
    df: pd.DataFrame,
    label_column: str,
) -> Dict:

    """
    Inspect class distribution for a single task.
    """

    _validate_dataframe(df)

    if label_column not in df.columns:
        raise ValueError(f"Column '{label_column}' not found")

    counts = df[label_column].value_counts(dropna=True).to_dict()

    logger.info(
        "Class distribution for '%s': %s",
        label_column,
        counts,
    )

    return counts


# -------------------------------------------------
# Multi-Task Distribution Inspection
# -------------------------------------------------

def check_multitask_distribution(
    df: pd.DataFrame,
    label_columns: List[str],
) -> Dict[str, Dict]:

    """
    Inspect class distribution for multiple tasks.
    """

    results = {}

    for col in label_columns:

        if col not in df.columns:
            logger.warning("Column '%s' missing. Skipping.", col)
            continue

        results[col] = check_class_distribution(df, col)

    return results


# -------------------------------------------------
# Random Oversampling
# -------------------------------------------------

def random_oversample(
    df: pd.DataFrame,
    label_column: str,
    random_state: int = 42,
) -> pd.DataFrame:

    """
    Balance dataset using random oversampling.
    """

    _validate_dataframe(df)

    if label_column not in df.columns:
        raise ValueError(f"Column '{label_column}' not found")

    df = df.dropna(subset=[label_column])

    counts = df[label_column].value_counts()

    if len(counts) < 2:
        logger.warning(
            "Only one class present for '%s'. Skipping oversampling.",
            label_column,
        )
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
        "Oversampled '%s' distribution: %s",
        label_column,
        balanced_df[label_column].value_counts().to_dict(),
    )

    return balanced_df


# -------------------------------------------------
# Random Undersampling
# -------------------------------------------------

def random_undersample(
    df: pd.DataFrame,
    label_column: str,
    random_state: int = 42,
) -> pd.DataFrame:

    """
    Balance dataset using random undersampling.
    """

    _validate_dataframe(df)

    if label_column not in df.columns:
        raise ValueError(f"Column '{label_column}' not found")

    df = df.dropna(subset=[label_column])

    counts = df[label_column].value_counts()

    if len(counts) < 2:
        logger.warning(
            "Only one class present for '%s'. Skipping undersampling.",
            label_column,
        )
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
        "Undersampled '%s' distribution: %s",
        label_column,
        balanced_df[label_column].value_counts().to_dict(),
    )

    return balanced_df


# -------------------------------------------------
# Automatic Balancing (Single Task)
# -------------------------------------------------

def balance_dataset(
    df: pd.DataFrame,
    label_column: str,
    method: str = "oversample",
    random_state: int = 42,
) -> pd.DataFrame:

    """
    Automatically balance a single task dataset.
    """

    if method == "oversample":

        return random_oversample(
            df,
            label_column,
            random_state=random_state,
        )

    elif method == "undersample":

        return random_undersample(
            df,
            label_column,
            random_state=random_state,
        )

    else:

        raise ValueError(
            "method must be 'oversample' or 'undersample'"
        )


# -------------------------------------------------
# Multi-Task Dataset Balancing
# -------------------------------------------------

def balance_multitask_dataset(
    df: pd.DataFrame,
    label_columns: List[str],
    method: str = "oversample",
    random_state: int = 42,
) -> pd.DataFrame:

    """
    Balance each task independently in a multi-task dataset.

    Useful for systems with multiple heads such as:

    bias
    ideology
    propaganda
    narrative
    emotion
    """

    _validate_dataframe(df)

    balanced_frames: List[pd.DataFrame] = []

    for label in label_columns:

        if label not in df.columns:

            logger.warning(
                "Label column '%s' missing. Skipping.",
                label,
            )
            continue

        task_df = df.dropna(subset=[label])

        if len(task_df) == 0:

            logger.warning(
                "No valid samples for '%s'. Skipping.",
                label,
            )
            continue

        logger.info(
            "Balancing task '%s' with %d samples",
            label,
            len(task_df),
        )

        balanced = balance_dataset(
            task_df,
            label_column=label,
            method=method,
            random_state=random_state,
        )

        balanced_frames.append(balanced)

    if not balanced_frames:
        raise ValueError("No tasks were balanced")

    combined = pd.concat(balanced_frames)

    combined = combined.drop_duplicates().reset_index(drop=True)

    logger.info("Final balanced dataset size: %d", len(combined))

    return combined