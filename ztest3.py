"""
File Name: split_allsides_dataset.py
Module: Ideology Detection - Dataset Preparation

Description
-----------
Creates train / validation / test splits for the AllSides Media Bias dataset
used in the ideology detection head of the TruthLens AI system.

This script assumes the dataset has already been cleaned and preprocessed.
It performs dataset validation, stratified splitting, and saves the splits
for downstream model training.

Dataset
-------
AllSides Media Bias Dataset (~65k samples)

Expected Input Columns
----------------------
title
text
label

Output Files
------------
allsides_train.csv
allsides_validation.csv
allsides_test.csv

Author: TruthLens AI
Date: 2026
Dependencies:
    pandas
    scikit-learn
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Tuple

import pandas as pd
from sklearn.model_selection import train_test_split

# -------------------------------------------------------
# PROJECT PATHS
# -------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent

INPUT_PATH = PROJECT_ROOT / "data" / "processed" / "ideology" / "allsides_processed2.csv"

OUTPUT_DIR = PROJECT_ROOT / "data" / "splits"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# -------------------------------------------------------
# CONFIGURATION
# -------------------------------------------------------

TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

RANDOM_SEED = 42

REQUIRED_COLUMNS = ["title", "text", "label"]

# -------------------------------------------------------
# LOGGER
# -------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

logger = logging.getLogger(__name__)

# -------------------------------------------------------
# DATASET VALIDATION
# -------------------------------------------------------

def validate_dataset(df: pd.DataFrame) -> None:
    """
    Ensure dataset contains required schema.
    """

    logger.info("Validating dataset schema...")

    missing_cols = [col for col in REQUIRED_COLUMNS if col not in df.columns]

    if missing_cols:
        raise ValueError(
            f"Dataset missing required columns: {missing_cols}"
        )

    logger.info("Dataset schema validation passed.")

# -------------------------------------------------------
# DATASET SPLIT
# -------------------------------------------------------

def split_dataset(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Perform stratified dataset split (70 / 15 / 15).
    """

    logger.info("Shuffling dataset...")

    df = df.sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)

    logger.info("Performing stratified split (70 / 15 / 15)...")

    train_df, temp_df = train_test_split(
        df,
        test_size=(1 - TRAIN_RATIO),
        stratify=df["label"],
        random_state=RANDOM_SEED,
    )

    val_df, test_df = train_test_split(
        temp_df,
        test_size=TEST_RATIO / (TEST_RATIO + VAL_RATIO),
        stratify=temp_df["label"],
        random_state=RANDOM_SEED,
    )

    logger.info(
        "Split sizes | Train: %d | Validation: %d | Test: %d",
        len(train_df),
        len(val_df),
        len(test_df),
    )

    return train_df, val_df, test_df

# -------------------------------------------------------
# LABEL DISTRIBUTION REPORT
# -------------------------------------------------------

def report_distribution(df: pd.DataFrame, name: str) -> None:
    """
    Log label distribution for a dataset split.
    """

    logger.info("Label distribution (%s):", name)

    distribution = df["label"].value_counts(normalize=True)

    for label, pct in distribution.items():
        logger.info("  %s: %.4f", label, pct)

# -------------------------------------------------------
# SAVE DATASETS
# -------------------------------------------------------

def save_splits(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> None:
    """
    Save dataset splits to disk.
    """

    train_path = OUTPUT_DIR / "allsides_train.csv"
    val_path = OUTPUT_DIR / "allsides_validation.csv"
    test_path = OUTPUT_DIR / "allsides_test.csv"

    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)
    test_df.to_csv(test_path, index=False)

    logger.info("Saved dataset splits:")
    logger.info("  %s", train_path)
    logger.info("  %s", val_path)
    logger.info("  %s", test_path)

# -------------------------------------------------------
# MAIN PIPELINE
# -------------------------------------------------------

def main() -> None:
    """
    Execute dataset splitting pipeline.
    """

    logger.info("Loading dataset...")
    logger.info("Input path: %s", INPUT_PATH)

    if not INPUT_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {INPUT_PATH}")

    df = pd.read_csv(INPUT_PATH)

    logger.info("Total samples loaded: %d", len(df))

    # Validate schema
    validate_dataset(df)

    # Perform split
    train_df, val_df, test_df = split_dataset(df)

    # Distribution reports
    report_distribution(train_df, "TRAIN")
    report_distribution(val_df, "VALIDATION")
    report_distribution(test_df, "TEST")

    # Save datasets
    save_splits(train_df, val_df, test_df)

    logger.info("Dataset split pipeline completed successfully.")

# -------------------------------------------------------
# ENTRY POINT
# -------------------------------------------------------

if __name__ == "__main__":
    main()