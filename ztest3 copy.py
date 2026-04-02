"""
File Name: build_unified_dataset.py
Module: Multi-Task Dataset Builder

Description
-----------
Build a unified multi-task dataset for the TruthLens AI system by combining
all task-specific split files into one canonical schema.

Canonical Tasks
---------------
1. Bias detection (bias_label)
2. Ideology detection (ideology_label)
3. Propaganda detection (propaganda_label)
4. Frame classification (frame)
5. Narrative role extraction (hero, villain, victim)
6. Narrative frame detection (CO, EC, HI, MO, RE)
7. Emotion detection (emotion_0 ... emotion_19)

Output
------
data/unified_dataset_<split>.csv
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import List

import pandas as pd

# ---------------------------------------------------------
# PATH CONFIGURATION
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_SPLITS_DIR = PROJECT_ROOT / "data" / "splits"

# ---------------------------------------------------------
# LOGGER CONFIGURATION
# ---------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

logger = logging.getLogger("dataset_builder")

# ---------------------------------------------------------
# SCHEMA
# ---------------------------------------------------------

EMOTION_COLUMNS = [f"emotion_{i}" for i in range(20)]

MASTER_COLUMNS = [
    "title",
    "text",
    "bias_label",
    "ideology_label",
    "propaganda_label",
    "frame",
    "CO",
    "EC",
    "HI",
    "MO",
    "RE",
    "hero",
    "villain",
    "victim",
    "hero_entities",
    "villain_entities",
    "victim_entities",
    *EMOTION_COLUMNS,
    "dataset",
]

_COLUMN_ALIASES = {
    "bias": "bias_label",
    "ideology": "ideology_label",
    "propaganda": "propaganda_label",
    "idealogy_label": "ideology_label",
    "co": "CO",
    "ec": "EC",
    "hi": "HI",
    "mo": "MO",
    "re": "RE",
    "narrative_hero": "hero",
    "narrative_villain": "villain",
    "narrative_victim": "victim",
    "dataset_source": "dataset",
}

_SPLIT_FILES = {
    "babe": "babe_{split}.csv",
    "basil": "basil_{split}.csv",
    "mbic": "mbic_{split}.csv",
    "allsides": "allsides_{split}.csv",
    "goemotion": "goemotion_{split}.csv",
    "semeval": "semeval_emotions_{split}.csv",
    "propaganda": "propaganda_{split}.csv",
    "framenet": "framenet_{split}.csv",
    "narrative": "narrative_{split}.csv",
}

_VALID_SPLITS = {"train", "validation", "test"}


# ---------------------------------------------------------
# UTILITIES
# ---------------------------------------------------------


def validate_file(path: Path) -> None:
    if not path.exists():
        logger.error("Dataset not found: %s", path)
        raise FileNotFoundError(path)


def load_csv_safe(path: Path) -> pd.DataFrame:
    validate_file(path)

    try:
        df = pd.read_csv(path)
    except Exception as exc:
        logger.error("Failed loading %s | %s", path, exc)
        raise

    logger.info("Loaded %s | rows=%s", path.name, len(df))
    return df


def _apply_aliases(df: pd.DataFrame) -> pd.DataFrame:
    normalized = df.copy()

    for alias, canonical in _COLUMN_ALIASES.items():
        if alias in normalized.columns and canonical not in normalized.columns:
            normalized[canonical] = normalized[alias]

    return normalized


def normalize_text_columns(df: pd.DataFrame) -> pd.DataFrame:
    if "text" not in df.columns:
        raise ValueError("Dataset missing required column: text")

    normalized = df.copy()

    if "title" not in normalized.columns:
        normalized["title"] = ""

    normalized["title"] = normalized["title"].fillna("").astype(str)
    normalized["text"] = normalized["text"].fillna("").astype(str)

    return normalized


def standardize_schema(df: pd.DataFrame, dataset_name: str) -> pd.DataFrame:
    normalized = _apply_aliases(df)
    normalized = normalize_text_columns(normalized)

    normalized["dataset"] = dataset_name

    for col in MASTER_COLUMNS:
        if col not in normalized.columns:
            normalized[col] = pd.NA

    return normalized[MASTER_COLUMNS]


# ---------------------------------------------------------
# DATASET LOADERS
# ---------------------------------------------------------


def _split_file(data_dir: Path, split: str, dataset_key: str) -> Path:
    filename = _SPLIT_FILES[dataset_key].format(split=split)
    return data_dir / split / filename


def load_bias_dataset(path: Path, name: str) -> pd.DataFrame:
    df = load_csv_safe(path)
    return standardize_schema(df, name)


def load_allsides(path: Path) -> pd.DataFrame:
    df = load_csv_safe(path)
    return standardize_schema(df, "allsides")


def load_emotion_dataset(path: Path, name: str) -> pd.DataFrame:
    df = load_csv_safe(path)

    for col in EMOTION_COLUMNS:
        if col not in df.columns:
            df[col] = pd.NA

    return standardize_schema(df, name)


def load_propaganda(path: Path) -> pd.DataFrame:
    df = load_csv_safe(path)
    return standardize_schema(df, "propaganda")


def load_framenet(path: Path) -> pd.DataFrame:
    df = load_csv_safe(path)
    return standardize_schema(df, "framenet")


def load_narrative(path: Path) -> pd.DataFrame:
    df = load_csv_safe(path)
    return standardize_schema(df, "narrative")


# ---------------------------------------------------------
# PIPELINE
# ---------------------------------------------------------


def build_unified_dataset(split: str, data_dir: Path = DEFAULT_SPLITS_DIR) -> pd.DataFrame:
    if split not in _VALID_SPLITS:
        raise ValueError(f"split must be one of {_VALID_SPLITS}, got: {split}")

    logger.info("Starting unified dataset build | split=%s", split)

    datasets: List[pd.DataFrame] = [
        load_bias_dataset(_split_file(data_dir, split, "babe"), "babe"),
        load_bias_dataset(_split_file(data_dir, split, "basil"), "basil"),
        load_bias_dataset(_split_file(data_dir, split, "mbic"), "mbic"),
        load_allsides(_split_file(data_dir, split, "allsides")),
        load_emotion_dataset(_split_file(data_dir, split, "goemotion"), "goemotion"),
        load_emotion_dataset(_split_file(data_dir, split, "semeval"), "semeval"),
        load_propaganda(_split_file(data_dir, split, "propaganda")),
        load_framenet(_split_file(data_dir, split, "framenet")),
        load_narrative(_split_file(data_dir, split, "narrative")),
    ]

    unified = pd.concat(datasets, ignore_index=True)

    logger.info("Unified dataset size (%s): %s", split, len(unified))

    return unified


def save_dataset(df: pd.DataFrame, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    logger.info("Unified dataset saved to %s", output_path)


# ---------------------------------------------------------
# CLI
# ---------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build canonical unified dataset split")

    parser.add_argument(
        "--split",
        choices=sorted(_VALID_SPLITS),
        default="validation",
        help="Split name to build",
    )
    parser.add_argument(
        "--splits-dir",
        default=str(DEFAULT_SPLITS_DIR),
        help="Root directory containing train/validation/test split folders",
    )
    parser.add_argument(
        "--output-path",
        default=None,
        help="Optional output path. Defaults to data/unified_dataset_<split>.csv",
    )

    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    split = str(args.split)
    splits_dir = Path(args.splits_dir).resolve()

    if args.output_path:
        output_path = Path(args.output_path).resolve()
    else:
        output_path = PROJECT_ROOT / "data" / f"unified_dataset_{split}.csv"

    unified = build_unified_dataset(split=split, data_dir=splits_dir)

    if unified.empty:
        raise RuntimeError("Unified dataset build produced zero rows")

    save_dataset(unified, output_path)


if __name__ == "__main__":
    main()
