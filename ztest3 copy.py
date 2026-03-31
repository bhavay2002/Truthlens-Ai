"""
File: split_emotion_dataset.py

Purpose
-------
Convert comma-separated emotion labels to multi-hot format
and split dataset into Train / Validation / Test.

Dataset
-------
GoEmotions subset (0–19 emotions)

Output
------
emotion_train.csv
emotion_validation.csv
emotion_test.csv
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


# -------------------------------------------------------
# PROJECT PATHS
# -------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent

INPUT_PATH = PROJECT_ROOT / "data" / "processed" / "emotion" / "semeval_emotions_processed2.csv"

OUTPUT_DIR = PROJECT_ROOT / "data" / "splits"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# -------------------------------------------------------
# CONFIG
# -------------------------------------------------------

NUM_EMOTIONS = 20

TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

RANDOM_SEED = 42


# -------------------------------------------------------
# LOGGER
# -------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

logger = logging.getLogger(__name__)


# -------------------------------------------------------
# MULTI-LABEL CONVERSION
# -------------------------------------------------------

def convert_to_multilabel(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert comma-separated emotion labels into binary columns.
    """

    logger.info("Converting labels to multi-hot format...")

    # Create columns
    for i in range(NUM_EMOTIONS):
        df[f"emotion_{i}"] = 0

    # Fill labels
    for idx, labels in df["labels"].items():

        if pd.isna(labels):
            continue

        labels = str(labels).split(",")

        for lab in labels:
            lab = lab.strip()

            if lab.isdigit():
                lab = int(lab)

                if 0 <= lab < NUM_EMOTIONS:
                    df.at[idx, f"emotion_{lab}"] = 1

    df = df.drop(columns=["labels"])

    return df


# -------------------------------------------------------
# DATASET SPLIT
# -------------------------------------------------------

def split_dataset(df: pd.DataFrame):

    logger.info("Splitting dataset (70 / 15 / 15)...")

    train_df, temp_df = train_test_split(
        df,
        test_size=(1 - TRAIN_RATIO),
        random_state=RANDOM_SEED,
        shuffle=True,
    )

    val_df, test_df = train_test_split(
        temp_df,
        test_size=TEST_RATIO / (TEST_RATIO + VAL_RATIO),
        random_state=RANDOM_SEED,
        shuffle=True,
    )

    return train_df, val_df, test_df


# -------------------------------------------------------
# LABEL DISTRIBUTION REPORT
# -------------------------------------------------------

def print_distribution(df: pd.DataFrame, name: str):

    logger.info(f"\n{name} emotion distribution:")

    emotion_cols = [f"emotion_{i}" for i in range(NUM_EMOTIONS)]

    counts = df[emotion_cols].sum()

    total = len(df)

    for i, count in counts.items():
        pct = count / total
        logger.info(f"{i}: {pct:.4f}")


# -------------------------------------------------------
# MAIN PIPELINE
# -------------------------------------------------------

def main():

    logger.info("Loading dataset...")
    logger.info(INPUT_PATH)

    df = pd.read_csv(INPUT_PATH)

    logger.info(f"Total samples: {len(df)}")

    # Remove duplicates
    if "text" in df.columns:
        df = df.drop_duplicates(subset="text")

    # Convert labels
    df = convert_to_multilabel(df)

    # Split
    train_df, val_df, test_df = split_dataset(df)

    logger.info("Dataset split completed.")

    print_distribution(train_df, "TRAIN")
    print_distribution(val_df, "VALIDATION")
    print_distribution(test_df, "TEST")

    # Save files
    train_path = OUTPUT_DIR / "semeval_emotions_train.csv"
    val_path = OUTPUT_DIR / "semeval_emotions_validation.csv"
    test_path = OUTPUT_DIR / "semeval_emotions_test.csv"

    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)
    test_df.to_csv(test_path, index=False)

    logger.info("\nFiles saved:")
    logger.info(train_path)
    logger.info(val_path)
    logger.info(test_path)


# -------------------------------------------------------
# ENTRY POINT
# -------------------------------------------------------

if __name__ == "__main__":
    main()