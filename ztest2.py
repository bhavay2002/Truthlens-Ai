from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd

from src.data.clean_data2 import clean_dataframe
from src.data.data_profiler import profile_dataset
from src.data.eda import run_eda


# --------------------------------------------------
# PATHS
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent

DEFAULT_DATA_PATH = PROJECT_ROOT / "data" / "interim" / "framenet_full.csv"

DEFAULT_OUTPUT_PATH = (
    PROJECT_ROOT / "data" / "processed" / "framenet_processed2.csv"
)


# --------------------------------------------------
# LOAD DATASET
# --------------------------------------------------

def load_dataset(path: Path) -> pd.DataFrame:

    # Single CSV
    if path.is_file():
        print("Loading dataset file:", path)
        df = pd.read_csv(path)
        print("Rows loaded:", len(df))
        return df

    # Folder with CSV files
    if path.is_dir():

        csv_files = list(path.glob("*.csv"))

        if not csv_files:
            raise FileNotFoundError(
                f"No CSV files found in directory: {path}"
            )

        dfs = []

        for f in csv_files:
            print("Loading:", f.name)
            dfs.append(pd.read_csv(f))

        df = pd.concat(dfs, ignore_index=True)

        print("Rows loaded:", len(df))

        return df

    raise FileNotFoundError(f"Invalid dataset path: {path}")


# --------------------------------------------------
# NORMALIZE FRAMENET DATA
# --------------------------------------------------

def normalize_framenet(df: pd.DataFrame) -> pd.DataFrame:

    required_cols = {"text", "frame"}

    if not required_cols.issubset(df.columns):
        print("FrameNet columns not detected — skipping normalization")
        return df

    print("Converting FrameNet annotations → sentence-frame dataset")

    df = df[["text", "frame", "frame_element"]].copy()

    df["text"] = df["text"].astype(str).str.strip()
    df["frame"] = df["frame"].astype(str).str.lower()

    df = df[df["text"].str.len() > 3]
    df = df[df["frame"].notna()]

    df["frame_element"] = df["frame_element"].fillna("")

    grouped = df.groupby(["text", "frame"])

    rows = []

    for (text, frame), g in grouped:

        elements = list(set(g["frame_element"].dropna()))

        rows.append({
            "text": text,
            "frame": frame,
            "frame_elements": ",".join(elements)
        })

    df_new = pd.DataFrame(rows)

    print("Normalized rows:", len(df_new))

    return df_new


# --------------------------------------------------
# NARRATIVE ROLE GENERATION
# --------------------------------------------------

def add_narrative_roles(df: pd.DataFrame) -> pd.DataFrame:

    villain_frames = {
        "attack",
        "cause_harm",
        "destroying",
        "killing",
        "hostile_encounter",
        "violence"
    }

    hero_frames = {
        "helping",
        "protecting",
        "rescuing",
        "supporting",
        "assistance",
        "defending"
    }

    victim_frames = {
        "victimization",
        "suffering",
        "harm"
    }

    df["frame"] = df["frame"].fillna("").str.lower()

    df["hero"] = df["frame"].apply(
        lambda x: int(any(f in x for f in hero_frames))
    )

    df["villain"] = df["frame"].apply(
        lambda x: int(any(f in x for f in villain_frames))
    )

    df["victim"] = df["frame"].apply(
        lambda x: int(any(f in x for f in victim_frames))
    )

    return df


# --------------------------------------------------
# MAIN PIPELINE
# --------------------------------------------------

def run(
    data_path: Path = DEFAULT_DATA_PATH,
    output_path: Path = DEFAULT_OUTPUT_PATH,
    skip_eda: bool = False,
) -> Path:

    print("Dataset path:", data_path)

    df = load_dataset(data_path)

    # --------------------------------------------------
    # PROFILE
    # --------------------------------------------------

    profile_results = profile_dataset(str(data_path))

    print("Profile keys:", list(profile_results.keys()))

    # --------------------------------------------------
    # EDA
    # --------------------------------------------------

    if not skip_eda:
        print("Running EDA...")
        run_eda(str(data_path))

    # --------------------------------------------------
    # NORMALIZE FRAMENET
    # --------------------------------------------------

    df = normalize_framenet(df)

    # --------------------------------------------------
    # REMOVE DUPLICATES
    # --------------------------------------------------

    df = df.drop_duplicates(subset=["text", "frame"])

    print("Rows after duplicate removal:", len(df))

    # --------------------------------------------------
    # TEXT CLEANING
    # --------------------------------------------------

    df = clean_dataframe(df, text_column="text")

    print("Rows after cleaning:", len(df))

    # --------------------------------------------------
    # GENERATE NARRATIVE LABELS
    # --------------------------------------------------

    df = add_narrative_roles(df)

    # --------------------------------------------------
    # SHUFFLE DATA
    # --------------------------------------------------

    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    # --------------------------------------------------
    # SAVE
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
        description="FrameNet Narrative Processing Pipeline"
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