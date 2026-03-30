"""
File Name: convert_bias_labels_inplace.py
Module: Dataset Preprocessing - Bias Label Conversion
Description:
    Converts bias labels in the dataset to binary format and writes
    the changes back to the original dataset file.

    Label Mapping:
        Biased      -> 1
        Non-biased  -> 0

    Any other labels (e.g., "No agreement") are removed.

Author: Your Name
Date: 2026-03-30

Dependencies:
    pandas
    pathlib

Inputs:
    CSV dataset containing columns:
        text
        label

Outputs:
    The same dataset file overwritten with binary labels.
"""

import pandas as pd
from pathlib import Path


# -------------------------
# Project Path Configuration
# -------------------------

PROJECT_ROOT = Path(__file__).resolve().parent

DEFAULT_DATA_PATH = PROJECT_ROOT / "data" / "processed" /"bias" / "babe_dataset_processed.csv"


# -------------------------
# Label Conversion Function
# -------------------------

def convert_bias_labels_inplace(file_path: Path):
    """
    Convert textual bias labels to binary and overwrite dataset.
    """

    print(f"Loading dataset from: {file_path}")

    df = pd.read_csv(file_path)

    print(f"Original dataset size: {len(df)}")

    # Label mapping
    label_map = {
        "Biased": 1,
        "Non-biased": 0
    }

    # Keep only valid labels
    df = df[df["label"].isin(label_map.keys())]

    # Convert labels
    df["label"] = df["label"].map(label_map)

    # Reset index
    df = df.reset_index(drop=True)

    print("Label distribution after conversion:")
    print(df["label"].value_counts())

    # Overwrite original dataset
    df.to_csv(file_path, index=False)

    print("Dataset successfully updated in original file.")


# -------------------------
# Main Execution
# -------------------------

if __name__ == "__main__":
    convert_bias_labels_inplace(DEFAULT_DATA_PATH)