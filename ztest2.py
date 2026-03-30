from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd

from src.data.class_balance import balance_dataset
from src.data.clean_data import clean_dataframe
from src.data.data_augmentation import augment_dataset
from src.data.data_profiler import profile_dataset
from src.data.eda import run_eda
from src.data.validate_data import validate_dataset


# --------------------------------------------------
# PATHS
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent

DEFAULT_DATA_PATH = PROJECT_ROOT / "data" / "interim" / "framenet_all_frames_dataset2.csv"

DEFAULT_OUTPUT_PATH = PROJECT_ROOT / "data" / "processed" / "framenet_all_frames_dataset_processed2.csv"


# --------------------------------------------------
# NARRATIVE FRAME LABELS
# --------------------------------------------------

FRAME_COLUMNS = ["RE", "HI", "CO", "MO", "EC"]


# --------------------------------------------------
# FRAME → NARRATIVE MAPPING (FrameNet)
# --------------------------------------------------

FRAME_TO_NARRATIVE = {

    # conflict
    "Attack": "CO",
    "Hostile_encounter": "CO",
    "Killing": "CO",
    "Protest": "CO",

    # responsibility
    "Blame": "RE",
    "Responsibility": "RE",
    "Judgment": "RE",

    # morality
    "Justice": "MO",
    "Crime": "MO",
    "Punishment": "MO",

    # economy
    "Commerce_buy": "EC",
    "Commerce_sell": "EC",
    "Economic_activity": "EC",

}


# --------------------------------------------------
# LABEL NORMALIZATION
# --------------------------------------------------

def normalize_labels(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize labels across datasets.
    """

    # -----------------------------
    # Bias labels
    # -----------------------------

    if "label" in df.columns:

        df["label"] = df["label"].astype(str).str.lower().str.strip()

        label_map = {
            "biased": 1,
            "non-biased": 0,
            "neutral": 0,
            "lexical": 1,
            "informational": 1,
            "0": 0,
            "1": 1,
            "-1": 0,
            "left": 0,
            "center": 1,
            "right": 2,
        }

        df["label"] = df["label"].map(label_map).fillna(df["label"])
        df = df.dropna(subset=["label"])
        df["label"] = df["label"].astype(int)

    # -----------------------------
    # Propaganda labels
    # -----------------------------

    if "binary_label" in df.columns:
        df["binary_label"] = df["binary_label"].astype(int)

    if "technique_id" in df.columns:
        df["technique_id"] = df["technique_id"].astype(int)

    # -----------------------------
    # Narrative frame labels
    # -----------------------------

    for col in FRAME_COLUMNS:

        if col in df.columns:

            df[col] = df[col].astype(str).str.lower()

            frame_map = {
                "true": 1,
                "false": 0,
                "yes": 1,
                "no": 0,
                "1": 1,
                "0": 0,
            }

            df[col] = df[col].map(frame_map).fillna(0).astype(int)

    return df


# --------------------------------------------------
# FRAME MAPPING (FrameNet)
# --------------------------------------------------

def convert_framenet_frames(df: pd.DataFrame):

    if "frame" not in df.columns:
        return df

    for col in FRAME_COLUMNS:
        if col not in df.columns:
            df[col] = 0

    for frame, narrative_label in FRAME_TO_NARRATIVE.items():

        mask = df["frame"] == frame
        df.loc[mask, narrative_label] = 1

    return df


# --------------------------------------------------
# MAIN PIPELINE
# --------------------------------------------------

def run(
    data_path: Path = DEFAULT_DATA_PATH,
    output_path: Path = DEFAULT_OUTPUT_PATH,
    skip_eda: bool = False,
) -> Path:

    if not data_path.exists():
        raise FileNotFoundError(f"Dataset not found: {data_path}")

    print("Input dataset:", data_path)

    # --------------------------------------------------
    # Validation
    # --------------------------------------------------

    validation_results = validate_dataset(
        str(data_path),
        label_columns=["label", "binary_label", "technique_id"],
    )

    print("Validation passed:", validation_results.get("all_passed"))

    # --------------------------------------------------
    # Profiling
    # --------------------------------------------------

    profile_results = profile_dataset(
        str(data_path),
        label_columns=["label", "binary_label", "technique_id"],
    )

    print("Profile keys:", list(profile_results.keys()))

    # --------------------------------------------------
    # EDA
    # --------------------------------------------------

    if not skip_eda:
        print("Running EDA...")
        run_eda(str(data_path))

    # --------------------------------------------------
    # Load dataset
    # --------------------------------------------------

    df = pd.read_csv(data_path)

    print("Rows loaded:", len(df))

    # --------------------------------------------------
    # Normalize column names
    # --------------------------------------------------

    if "sentence" in df.columns and "text" not in df.columns:
        df = df.rename(columns={"sentence": "text"})

    if "text" not in df.columns:
        raise ValueError("Dataset must contain a 'text' column")

    # --------------------------------------------------
    # Convert FrameNet frames
    # --------------------------------------------------

    df = convert_framenet_frames(df)

    # --------------------------------------------------
    # Ensure narrative columns exist
    # --------------------------------------------------

    for col in FRAME_COLUMNS:
        if col not in df.columns:
            df[col] = 0

    # --------------------------------------------------
    # Basic cleaning
    # --------------------------------------------------

    df["text"] = df["text"].astype(str).str.strip()

    df = df[df["text"].str.len() > 10]

    df = df.drop_duplicates(subset=["text"])

    print("Rows after duplicate removal:", len(df))

    # --------------------------------------------------
    # Normalize labels
    # --------------------------------------------------

    df = normalize_labels(df)

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
    # Target column detection
    # --------------------------------------------------

    target_column = None

    if "label" in df.columns:
        target_column = "label"

    elif "technique_id" in df.columns:
        target_column = "technique_id"

    elif "binary_label" in df.columns:
        target_column = "binary_label"

    # Narrative frames → multi-label
    # Skip balancing

    # --------------------------------------------------
    # Class balancing
    # --------------------------------------------------

    if target_column and df[target_column].nunique() >= 2:

        df = balance_dataset(
            df,
            label_column=target_column,
            method="oversample",
        )

        print("Rows after balancing:", len(df))

    else:

        print("Skipping balancing (multi-label dataset)")

    # --------------------------------------------------
    # Data augmentation
    # --------------------------------------------------

    if len(df) > 0:

        df = augment_dataset(
            df,
            text_column="text",
            multiplier=2,
        )

        print("Rows after augmentation:", len(df))

    # --------------------------------------------------
    # Save dataset
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
        description="Run TruthLens preprocessing pipeline"
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