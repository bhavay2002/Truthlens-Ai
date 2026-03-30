from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd

from src.data.clean_data import clean_dataframe
from src.data.data_profiler import profile_dataset
from src.data.eda import run_eda
from src.data.validate_data import validate_dataset

from src.data.class_balance import balance_dataset
from src.data.data_augmentation import augment_dataset


# --------------------------------------------------
# PATHS
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent

DEFAULT_DATA_PATH = PROJECT_ROOT / "data" / "splits" / "propaganda2"

DEFAULT_OUTPUT_PATH = (
    PROJECT_ROOT / "data" / "processed" / "propaganda_processed.csv"
)


# --------------------------------------------------
# VALID LABELS
# --------------------------------------------------

VALID_LABELS = {0, 1}


# --------------------------------------------------
# LABEL VALIDATION
# --------------------------------------------------

def validate_labels(df: pd.DataFrame) -> pd.DataFrame:

    if "label" not in df.columns:
        raise ValueError("Dataset must contain a 'label' column")

    df = df[df["label"].isin(VALID_LABELS)]

    return df


# --------------------------------------------------
# LOAD SPLIT DATASETS
# --------------------------------------------------

def load_splits(data_dir: Path) -> pd.DataFrame:

    train_file = data_dir / "Train_preprocessed.csv"
    val_file = data_dir / "Valid_preprocessed.csv"
    test_file = data_dir / "Test_preprocessed.csv"

    if not train_file.exists():
        raise FileNotFoundError(f"Missing file: {train_file}")

    if not val_file.exists():
        raise FileNotFoundError(f"Missing file: {val_file}")

    if not test_file.exists():
        raise FileNotFoundError(f"Missing file: {test_file}")

    print("Loading dataset splits...")

    train = pd.read_csv(train_file)
    val = pd.read_csv(val_file)
    test = pd.read_csv(test_file)

    print("Train size:", len(train))
    print("Validation size:", len(val))
    print("Test size:", len(test))

    merged = pd.concat([train, val, test], ignore_index=True)

    print("Merged dataset size:", len(merged))

    return merged


# --------------------------------------------------
# MAIN PIPELINE
# --------------------------------------------------

def run(
    data_path: Path = DEFAULT_DATA_PATH,
    output_path: Path = DEFAULT_OUTPUT_PATH,
    skip_eda: bool = False,
) -> Path:

    if not data_path.exists():
        raise FileNotFoundError(f"Dataset folder not found: {data_path}")

    print("Dataset folder:", data_path)

    # --------------------------------------------------
    # Load and merge datasets
    # --------------------------------------------------

    df = load_splits(data_path)

    # --------------------------------------------------
    # Profiling
    # --------------------------------------------------

    profile_results = profile_dataset(
        str(data_path / "Train_preprocessed.csv"),
        label_columns=["label"],
    )

    print("Profile keys:", list(profile_results.keys()))

    # --------------------------------------------------
    # EDA
    # --------------------------------------------------

    if not skip_eda:
        print("Running EDA...")
        run_eda(str(data_path / "Train_preprocessed.csv"))

    # --------------------------------------------------
    # Normalize text column
    # --------------------------------------------------

    if "sentence" in df.columns and "text" not in df.columns:
        df = df.rename(columns={"sentence": "text"})

    if "text" not in df.columns:
        raise ValueError("Dataset must contain a 'text' column")

    # --------------------------------------------------
    # Label validation
    # --------------------------------------------------

    df = validate_labels(df)

    print("Rows after label validation:", len(df))

    # --------------------------------------------------
    # Basic cleaning
    # --------------------------------------------------

    df["text"] = df["text"].astype(str).str.strip()

    df = df[df["text"].str.len() > 10]

    df = df.drop_duplicates(subset=["text"])

    print("Rows after duplicate removal:", len(df))

    # --------------------------------------------------
    # Transformer safety
    # --------------------------------------------------

    df["text"] = df["text"].str.slice(0, 2000)

    # --------------------------------------------------
    # Text cleaning
    # --------------------------------------------------

    df = clean_dataframe(df, text_column="text")

    print("Rows after cleaning:", len(df))

    # --------------------------------------------------
    # CLASS BALANCING
    # --------------------------------------------------

    print("Balancing dataset...")

    df = balance_dataset(
        df,
        label_column="label",
        method="oversample",
    )

    # --------------------------------------------------
    # AUGMENT PROPAGANDA CLASS
    # --------------------------------------------------

    print("Augmenting propaganda samples...")

    propaganda_df = df[df["label"] == 1]

    propaganda_aug = augment_dataset(
        propaganda_df,
        text_column="text",
        multiplier=2,
    )

    non_prop_df = df[df["label"] == 0]

    df = pd.concat(
        [non_prop_df, propaganda_aug],
        ignore_index=True,
    )

    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    print("Dataset size after augmentation:", len(df))

    # --------------------------------------------------
    # SAVE DATASET
    # --------------------------------------------------

    output_path.parent.mkdir(parents=True, exist_ok=True)

    df.to_csv(output_path, index=False)

    print("Final dataset size:", len(df))
    print("Saved to:", output_path)

    return output_path


# --------------------------------------------------
# CLI
# --------------------------------------------------

def _parse_args():

    parser = argparse.ArgumentParser(
        description="Run Propaganda preprocessing pipeline"
    )

    parser.add_argument(
        "--data-path",
        default=str(DEFAULT_DATA_PATH),
    )

    parser.add_argument(
        "--output-path",
        default=str(DEFAULT_OUTPUT_PATH),
    )

    parser.add_argument(
        "--skip-eda",
        action="store_true",
    )

    return parser.parse_args()


if __name__ == "__main__":

    args = _parse_args()

    run(
        data_path=Path(args.data_path),
        output_path=Path(args.output_path),
        skip_eda=args.skip_eda,
    )