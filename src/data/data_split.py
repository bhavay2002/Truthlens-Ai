"""
File: src/data/data_split.py

Purpose
-------
Dataset splitting utilities for machine learning pipelines.

Creates reproducible train / validation / test splits using
stratified sampling to preserve label distribution.

Typical Usage
-------------
Used after dataset cleaning and augmentation.

Inputs
------
df : pandas.DataFrame
    Dataset containing text and label columns.

Outputs
-------
split_dataset() -> (train_df, val_df, test_df)

save_splits() -> CSV files

Dependencies
------------
pandas
sklearn
pathlib
logging
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Tuple

import pandas as pd
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)


# -------------------------------------------------
# Dataset Splitting
# -------------------------------------------------

def split_dataset(
    df: pd.DataFrame,
    label_column: str = "label",
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    stratified: bool = True,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split dataset into train, validation, and test sets.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataset.

    label_column : str
        Column used for stratification.

    train_ratio : float
        Training dataset ratio.

    val_ratio : float
        Validation dataset ratio.

    test_ratio : float
        Test dataset ratio.

    random_state : int
        Seed for reproducibility.

    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]
        train_df, val_df, test_df
    """

    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame")

    if len(df) < 3:
        raise ValueError("Dataset must contain at least 3 rows")

    if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-6:
        raise ValueError(
            "Train, validation, and test ratios must sum to 1"
        )

    if train_ratio <= 0 or val_ratio <= 0 or test_ratio <= 0:
        raise ValueError("All split ratios must be > 0")

    if label_column not in df.columns:
        raise ValueError(
            f"Label column '{label_column}' not found in dataset"
        )

    logger.info("Starting dataset split")

    use_stratified = stratified
    if use_stratified:
        label_counts = df[label_column].value_counts()
        if len(label_counts) < 2 or label_counts.min() < 2:
            logger.warning(
                "Insufficient class counts for stratified split; falling back to random split"
            )
            use_stratified = False

    stratify_values = df[label_column] if use_stratified else None

    # -------------------------------------------------
    # Train + Temp split
    # -------------------------------------------------

    try:
        train_df, temp_df = train_test_split(
            df,
            test_size=(1 - train_ratio),
            stratify=stratify_values,
            random_state=random_state,
        )
    except ValueError as error:
        logger.warning("Stratified split failed (%s); retrying without stratification", error)
        train_df, temp_df = train_test_split(
            df,
            test_size=(1 - train_ratio),
            stratify=None,
            random_state=random_state,
        )

    # -------------------------------------------------
    # Validation + Test split
    # -------------------------------------------------

    val_size_adjusted = val_ratio / (val_ratio + test_ratio)

    temp_stratify = temp_df[label_column] if use_stratified else None

    try:
        val_df, test_df = train_test_split(
            temp_df,
            test_size=(1 - val_size_adjusted),
            stratify=temp_stratify,
            random_state=random_state,
        )
    except ValueError as error:
        logger.warning(
            "Validation/test stratified split failed (%s); retrying without stratification",
            error,
        )
        val_df, test_df = train_test_split(
            temp_df,
            test_size=(1 - val_size_adjusted),
            stratify=None,
            random_state=random_state,
        )

    logger.info(
        "Dataset split complete: train=%s, val=%s, test=%s",
        len(train_df),
        len(val_df),
        len(test_df),
    )

    return train_df.reset_index(drop=True), \
        val_df.reset_index(drop=True), \
        test_df.reset_index(drop=True)


# -------------------------------------------------
# Save Dataset Splits
# -------------------------------------------------

def save_splits(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    output_dir: str | Path = "data/splits",
) -> None:
    """
    Save dataset splits to CSV files.

    Parameters
    ----------
    train_df : pd.DataFrame
    val_df : pd.DataFrame
    test_df : pd.DataFrame
    output_dir : str | Path
    """

    output_dir = Path(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    train_path = output_dir / "train.csv"
    val_path = output_dir / "val.csv"
    test_path = output_dir / "test.csv"

    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)
    test_df.to_csv(test_path, index=False)

    logger.info("Saved dataset splits to %s", output_dir)


# -------------------------------------------------
# Load Dataset Splits
# -------------------------------------------------

def load_splits(
    input_dir: str | Path = "data/splits",
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load saved dataset splits.

    Parameters
    ----------
    input_dir : str | Path

    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]
    """

    input_dir = Path(input_dir)

    train_path = input_dir / "train.csv"
    val_path = input_dir / "val.csv"
    test_path = input_dir / "test.csv"

    if not train_path.exists():
        raise FileNotFoundError(f"Missing file: {train_path}")

    if not val_path.exists():
        raise FileNotFoundError(f"Missing file: {val_path}")

    if not test_path.exists():
        raise FileNotFoundError(f"Missing file: {test_path}")

    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)
    test_df = pd.read_csv(test_path)

    logger.info("Loaded dataset splits from %s", input_dir)

    return train_df, val_df, test_df
