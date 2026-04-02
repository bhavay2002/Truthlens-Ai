"""
File: src/pipelines/data_pipeline.py

Purpose
-------
End-to-end data pipeline for TruthLens AI.

Pipeline stages:
1. Load datasets
2. Validate dataset
3. Profile dataset
4. Clean text
5. Balance classes
6. Data augmentation
7. Train/validation/test split
8. Save processed datasets

Inputs
------
configs/data_config.yaml
datasets

Outputs
-------
processed dataset
train/val/test splits
EDA + profiling reports
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.load_data import merge_datasets, load_csv, normalize_schema
from src.data.validate_data import DataValidator
from src.data.data_profiler import DataProfiler
from src.data.clean_data import clean_dataframe
from src.data.class_balance import balance_dataset
from src.data.data_split import split_dataset, save_splits

logger = logging.getLogger(__name__)
DEFAULT_DATA_CONFIG_PATH = PROJECT_ROOT / "config" / "data_config.yaml"


def _resolve_project_path(path_value: str | Path) -> Path:
    path_obj = Path(path_value)
    if path_obj.is_absolute():
        return path_obj
    return (PROJECT_ROOT / path_obj).resolve()


def _require_keys(config: dict[str, Any], section: str, keys: tuple[str, ...]) -> None:
    if section not in config or not isinstance(config[section], dict):
        raise KeyError(f"Missing required config section: '{section}'")

    missing = [key for key in keys if key not in config[section]]
    if missing:
        raise KeyError(f"Missing keys in '{section}': {missing}")


def _validate_config(config: dict[str, Any]) -> None:
    _require_keys(
        config,
        "dataset",
        (
            "raw_data_dir",
            "text_column",
            "title_column",
            "label_column",
        ),
    )
    dataset_cfg = config["dataset"]
    has_pair_files = (
        "fake_news_file" in dataset_cfg
        and "real_news_file" in dataset_cfg
    )
    has_unified_file = "unified_dataset_file" in dataset_cfg
    if not has_pair_files and not has_unified_file:
        raise KeyError(
            "dataset config must include either "
            "('fake_news_file' and 'real_news_file') or 'unified_dataset_file'"
        )
    _require_keys(
        config,
        "validation",
        ("required_columns", "max_null_ratio", "max_duplicate_ratio", "min_class_ratio", "min_text_length"),
    )
    _require_keys(config, "cleaning", ("min_word_count",))
    _require_keys(config, "balancing", ("enabled", "method", "random_state"))
    _require_keys(config, "augmentation", ("enabled", "multiplier"))
    _require_keys(
        config,
        "split",
        ("train_ratio", "validation_ratio", "test_ratio", "random_state"),
    )
    _require_keys(config, "profiling", ("enabled", "report_dir"))
    _require_keys(config, "output", ("processed_data_dir", "splits_dir"))

    split_cfg = config["split"]
    ratio_sum = (
        float(split_cfg["train_ratio"])
        + float(split_cfg["validation_ratio"])
        + float(split_cfg["test_ratio"])
    )
    if abs(ratio_sum - 1.0) > 1e-6:
        raise ValueError("split ratios must sum to 1.0")


# --------------------------------------------------
# Load Config
# --------------------------------------------------

def load_config(config_path: str | Path = DEFAULT_DATA_CONFIG_PATH) -> dict[str, Any]:
    config_path = _resolve_project_path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as config_file:
        config = yaml.safe_load(config_file) or {}

    _validate_config(config)

    return config


# --------------------------------------------------
# Data Pipeline
# --------------------------------------------------

def run_data_pipeline(config_path: str | Path = DEFAULT_DATA_CONFIG_PATH) -> dict[str, Any]:

    logger.info("Starting data pipeline")

    config = load_config(config_path)

    dataset_cfg = config["dataset"]
    raw_data_dir = _resolve_project_path(dataset_cfg["raw_data_dir"])

    # --------------------------------------------------
    # Load Dataset
    # --------------------------------------------------

    if "unified_dataset_file" in dataset_cfg:
        unified_path = raw_data_dir / dataset_cfg["unified_dataset_file"]
        df = load_csv(unified_path)
        label_columns = dataset_cfg.get("label_columns")
        if not isinstance(label_columns, list) or not label_columns:
            label_columns = [dataset_cfg["label_column"]]

        df = normalize_schema(
            df,
            text_column=dataset_cfg["text_column"],
            title_column=dataset_cfg["title_column"],
            label_columns=label_columns,
        )
        if dataset_cfg["label_column"] not in df.columns:
            df[dataset_cfg["label_column"]] = pd.NA
    else:
        labels_cfg = dataset_cfg.get("labels", {})
        fake_label = int(labels_cfg.get("fake", 1))
        real_label = int(labels_cfg.get("real", 0))

        fake_path = raw_data_dir / dataset_cfg["fake_news_file"]
        real_path = raw_data_dir / dataset_cfg["real_news_file"]

        df = merge_datasets(
            fake_path,
            real_path,
            text_column=dataset_cfg["text_column"],
            title_column=dataset_cfg["title_column"],
            label_column=dataset_cfg["label_column"],
            fake_label=fake_label,
            real_label=real_label,
        )

    if df.empty:
        raise ValueError("Merged dataset is empty; cannot continue pipeline")

    # --------------------------------------------------
    # Validation
    # --------------------------------------------------

    configured_label_columns = dataset_cfg.get("label_columns")
    if isinstance(configured_label_columns, list) and configured_label_columns:
        label_columns = configured_label_columns
    else:
        label_columns = [dataset_cfg["label_column"]]

    label_specs = dataset_cfg.get("label_specs")
    if not isinstance(label_specs, dict):
        label_specs = None

    validator = DataValidator(
        required_columns=config["validation"]["required_columns"],
        label_columns=label_columns,
        label_specs=label_specs,
        max_null_ratio=config["validation"]["max_null_ratio"],
        max_dup_ratio=config["validation"]["max_duplicate_ratio"],
        min_class_ratio=config["validation"]["min_class_ratio"],
        min_text_length=config["validation"]["min_text_length"],
    )

    validation_results = validator.validate(df)

    if not validation_results["all_passed"]:
        logger.warning("Dataset validation reported issues")

    # --------------------------------------------------
    # Dataset Profiling
    # --------------------------------------------------

    if config["profiling"]["enabled"]:

        profiler = DataProfiler(
            df,
            text_column=dataset_cfg["text_column"],
            label_column=dataset_cfg["label_column"],
            report_dir=_resolve_project_path(config["profiling"]["report_dir"]),
        )

        profiler.profile()

    # --------------------------------------------------
    # Data Cleaning
    # --------------------------------------------------

    df = clean_dataframe(
        df,
        text_column=dataset_cfg["text_column"],
        title_column=dataset_cfg["title_column"],
        min_len=config["cleaning"]["min_word_count"],
    )

    # --------------------------------------------------
    # Class Balancing
    # --------------------------------------------------

    if config["balancing"]["enabled"]:

        df = balance_dataset(
            df,
            label_column=dataset_cfg["label_column"],
            method=config["balancing"]["method"],
            random_state=int(config["balancing"]["random_state"]),
        )

    # --------------------------------------------------
    # Data Augmentation
    # --------------------------------------------------

    if config["augmentation"]["enabled"]:
        from src.data.data_augmentation import augment_dataset

        df = augment_dataset(
            df,
            text_column=dataset_cfg["text_column"],
            multiplier=config["augmentation"]["multiplier"],
        )

    # --------------------------------------------------
    # Save Processed Dataset
    # --------------------------------------------------

    processed_dir = _resolve_project_path(config["output"]["processed_data_dir"])

    processed_dir.mkdir(parents=True, exist_ok=True)

    processed_path = processed_dir / "processed_dataset.csv"

    df.to_csv(processed_path, index=False)

    logger.info("Processed dataset saved: %s", processed_path)
    logger.info("Processed rows: %s", len(df))

    # --------------------------------------------------
    # Dataset Split
    # --------------------------------------------------

    train_df, val_df, test_df = split_dataset(
        df,
        label_column=dataset_cfg["label_column"],
        train_ratio=config["split"]["train_ratio"],
        val_ratio=config["split"]["validation_ratio"],
        test_ratio=config["split"]["test_ratio"],
        stratified=bool(config["split"].get("stratified", True)),
        random_state=config["split"]["random_state"],
    )

    splits_dir = _resolve_project_path(config["output"]["splits_dir"])

    save_splits(
        train_df,
        val_df,
        test_df,
        output_dir=splits_dir,
    )

    logger.info("Data pipeline completed successfully")

    return {
        "processed_dataset_path": str(processed_path),
        "splits_dir": str(splits_dir),
        "processed_rows": int(len(df)),
        "train_rows": int(len(train_df)),
        "val_rows": int(len(val_df)),
        "test_rows": int(len(test_df)),
    }


# --------------------------------------------------
# CLI Entry
# --------------------------------------------------

if __name__ == "__main__":

    logging.basicConfig(level=logging.INFO)

    run_data_pipeline()
