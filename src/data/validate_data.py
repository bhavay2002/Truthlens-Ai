"""
File: src/data/validate_data.py

Purpose
-------
Dataset validation utilities for NLP pipelines.
Ensures dataset quality before model training by checking schema integrity,
null values, duplicates, label distribution, text quality, and vocabulary size.

Typical Usage
-------------
Used before model training to verify dataset quality in pipelines
such as fake-news detection systems (e.g., TruthLens AI).

Inputs
------
df : pandas.DataFrame
    Dataset containing text and label columns.

csv_path : str
    Path to CSV dataset.

Outputs
-------
validate(df) -> Dict[str, Any]
validate_dataset(csv_path) -> Dict[str, Any]

Dependencies
------------
pandas
logging
typing
"""

from __future__ import annotations

import logging
from typing import List, Dict, Any

import pandas as pd

logger = logging.getLogger(__name__)


class DataValidator:
    """
    Validate dataset quality and structure before ML training.

    Parameters
    ----------
    required_columns : List[str]
        Columns that must exist in dataset.

    max_null_ratio : float
        Maximum allowed ratio of null values per column.

    max_dup_ratio : float
        Maximum allowed duplicate ratio.

    min_class_ratio : float
        Minimum ratio for each class to avoid severe imbalance.

    min_text_length : int
        Minimum allowed text length.
    """

    def __init__(
        self,
        required_columns: List[str] | None = None,
        max_null_ratio: float = 0.1,
        max_dup_ratio: float = 0.2,
        min_class_ratio: float = 0.2,
        min_text_length: int = 10,
    ):

        self.required_columns = required_columns or ["text", "label"]
        self.max_null_ratio = max_null_ratio
        self.max_dup_ratio = max_dup_ratio
        self.min_class_ratio = min_class_ratio
        self.min_text_length = min_text_length

        self.validation_errors: List[str] = []

    # ------------------------------------------------
    # Internal Helpers
    # ------------------------------------------------

    def _has_required_columns(self, df: pd.DataFrame) -> bool:
        """Check whether required columns exist."""
        return set(self.required_columns).issubset(df.columns)

    # ------------------------------------------------
    # Schema Validation
    # ------------------------------------------------

    def validate_schema(self, df: pd.DataFrame) -> bool:
        """
        Verify required columns exist in dataset.
        """

        missing_cols = set(self.required_columns) - set(df.columns)

        if missing_cols:
            error = f"Missing required columns: {missing_cols}"
            logger.error(error)
            self.validation_errors.append(error)
            return False

        return True

    # ------------------------------------------------
    # Null Validation
    # ------------------------------------------------

    def validate_nulls(
        self,
        df: pd.DataFrame,
        max_null_ratio: float | None = None,
    ) -> bool:
        """
        Detect excessive null values.
        """

        if not self._has_required_columns(df):

            error = "Cannot validate nulls: required columns missing"
            logger.error(error)
            self.validation_errors.append(error)
            return False

        threshold = self.max_null_ratio if max_null_ratio is None else max_null_ratio

        null_ratios = df[self.required_columns].isnull().mean()

        problematic = null_ratios[null_ratios > threshold]

        if not problematic.empty:

            error = f"Columns with excessive nulls: {problematic.to_dict()}"
            logger.warning(error)
            self.validation_errors.append(error)

            return False

        return True

    # ------------------------------------------------
    # Duplicate Validation
    # ------------------------------------------------

    def validate_duplicates(self, df: pd.DataFrame) -> bool:
        """
        Detect duplicate text samples.
        """

        if "text" not in df.columns:

            error = "Cannot validate duplicates: 'text' column missing"
            logger.error(error)
            self.validation_errors.append(error)

            return False

        if len(df) == 0:
            warning = "Cannot validate duplicates: dataframe is empty"
            logger.warning(warning)
            self.validation_errors.append(warning)
            return False

        dup_count = df.duplicated(subset=["text"]).sum()
        dup_ratio = dup_count / len(df)

        if dup_ratio > self.max_dup_ratio:

            warning = (
                f"High duplicate ratio detected: "
                f"{dup_ratio:.2%} ({dup_count} duplicates)"
            )

            logger.warning(warning)
            self.validation_errors.append(warning)

            return False

        return True

    # ------------------------------------------------
    # Label Distribution Validation
    # ------------------------------------------------

    def validate_labels(self, df: pd.DataFrame) -> bool:
        """
        Check class balance.
        """

        if "label" not in df.columns:
            return True

        if df["label"].nunique() < 2:
            warning = "Only one class present in label column"
            logger.warning(warning)
            self.validation_errors.append(warning)
            return False

        label_distribution = df["label"].value_counts(normalize=True)

        min_ratio = label_distribution.min()

        if min_ratio < self.min_class_ratio:

            warning = f"Class imbalance detected: {label_distribution.to_dict()}"

            logger.warning(warning)
            self.validation_errors.append(warning)

            return False

        return True

    # ------------------------------------------------
    # Text Quality Validation
    # ------------------------------------------------

    def validate_text_quality(self, df: pd.DataFrame) -> bool:
        """
        Ensure text samples are sufficiently long.
        """

        if "text" not in df.columns:
            return True

        if len(df) == 0:
            warning = "Cannot validate text quality: dataframe is empty"
            logger.warning(warning)
            self.validation_errors.append(warning)
            return False

        text_series = df["text"].astype(str)

        short_texts = (text_series.str.len() < self.min_text_length).sum()

        short_ratio = short_texts / len(df)

        if short_ratio > 0.1:

            warning = (
                f"Too many short texts detected: "
                f"{short_ratio:.2%} ({short_texts} samples "
                f"< {self.min_text_length} characters)"
            )

            logger.warning(warning)

            self.validation_errors.append(warning)

            return False

        return True

    # ------------------------------------------------
    # Vocabulary Validation
    # ------------------------------------------------

    def validate_vocabulary(self, df: pd.DataFrame) -> bool:
        """
        Check vocabulary diversity.
        """

        if "text" not in df.columns:
            return True

        words = " ".join(df["text"].astype(str)).split()

        vocab_size = len(set(words))

        if vocab_size < 50:

            warning = f"Very small vocabulary detected: {vocab_size} unique words"

            logger.warning(warning)

            self.validation_errors.append(warning)

            return False

        return True

    # ------------------------------------------------
    # Dataset Summary
    # ------------------------------------------------

    def dataset_summary(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Generate dataset statistics summary.
        """

        has_text = "text" in df.columns and len(df) > 0
        text_series = df["text"].astype(str) if "text" in df.columns else pd.Series(dtype=str)

        summary = {
            "rows": len(df),
            "columns": len(df.columns),
            "avg_text_length": int(text_series.str.len().mean()) if has_text else 0,
            "median_text_length": int(text_series.str.len().median()) if has_text else 0,
            "vocab_size": len(set(" ".join(text_series).split())) if has_text else 0,
        }

        if "label" in df.columns:

            summary["label_distribution"] = df["label"].value_counts().to_dict()

        return summary

    # ------------------------------------------------
    # Run Full Validation
    # ------------------------------------------------

    def validate(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Run complete dataset validation pipeline.
        """

        if len(df) == 0:
            raise ValueError("Dataset is empty")

        logger.info("Running dataset validation")

        self.validation_errors = []

        results = {
            "schema_valid": self.validate_schema(df),
            "nulls_valid": self.validate_nulls(df),
            "duplicates_valid": self.validate_duplicates(df),
            "labels_valid": self.validate_labels(df),
            "text_quality_valid": self.validate_text_quality(df),
            "vocabulary_valid": self.validate_vocabulary(df),
        }

        results["dataset_summary"] = self.dataset_summary(df)

        validation_flags = [
            results["schema_valid"],
            results["nulls_valid"],
            results["duplicates_valid"],
            results["labels_valid"],
            results["text_quality_valid"],
            results["vocabulary_valid"],
        ]

        all_passed = all(validation_flags)

        results["all_passed"] = all_passed
        results["errors"] = self.validation_errors

        if all_passed:
            logger.info("Dataset validation passed")
        else:
            logger.warning("Dataset validation issues detected")

            for err in self.validation_errors:
                logger.warning(f"- {err}")

        logger.info(f"Dataset summary: {results['dataset_summary']}")

        return results


# ------------------------------------------------
# Convenience Function
# ------------------------------------------------

def validate_dataset(csv_path: str) -> Dict[str, Any]:
    """
    Validate dataset directly from CSV file.

    Parameters
    ----------
    csv_path : str
        Path to dataset CSV.

    Returns
    -------
    Dict[str, Any]
        Validation results dictionary.
    """

    df = pd.read_csv(csv_path)

    validator = DataValidator()

    return validator.validate(df)
