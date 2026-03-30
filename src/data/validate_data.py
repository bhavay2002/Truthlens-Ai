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

    def __init__(
        self,
        required_columns: List[str] | None = None,
        label_columns: List[str] | None = None,
        label_specs: Dict[str, Dict[str, Any]] | None = None,
        max_null_ratio: float = 0.1,
        max_dup_ratio: float = 0.2,
        min_class_ratio: float = 0.05,
        min_text_length: int = 10,
    ):

        self.required_columns = required_columns or ["text"]

        self.label_columns = label_columns or []
        self.label_specs = label_specs or {}

        self.max_null_ratio = max_null_ratio
        self.max_dup_ratio = max_dup_ratio
        self.min_class_ratio = min_class_ratio
        self.min_text_length = min_text_length

        self.validation_errors: List[str] = []

    def _validate_label_spec(
        self,
        *,
        label: str,
        values: pd.Series,
    ) -> bool:
        """
        Validate values against an optional spec for a label column.

        Supported keys:
        - allowed_values: list
        - min_value: number
        - max_value: number
        """

        spec = self.label_specs.get(label)
        if not spec:
            return True

        results = True
        numeric = pd.to_numeric(values, errors="coerce")

        if numeric.isna().any():
            err = f"Non-numeric values detected in '{label}'"
            logger.warning(err)
            self.validation_errors.append(err)
            results = False
            return results

        allowed_values = spec.get("allowed_values")
        if isinstance(allowed_values, list) and allowed_values:
            allowed_set = set(allowed_values)
            invalid_mask = ~numeric.isin(allowed_set)
            if bool(invalid_mask.any()):
                invalid_values = sorted(set(numeric[invalid_mask].tolist()))
                err = (
                    f"Invalid values in '{label}': {invalid_values}. "
                    f"Allowed: {sorted(allowed_set)}"
                )
                logger.warning(err)
                self.validation_errors.append(err)
                results = False

        min_value = spec.get("min_value")
        if min_value is not None:
            if bool((numeric < float(min_value)).any()):
                err = f"Values below min_value={min_value} in '{label}'"
                logger.warning(err)
                self.validation_errors.append(err)
                results = False

        max_value = spec.get("max_value")
        if max_value is not None:
            if bool((numeric > float(max_value)).any()):
                err = f"Values above max_value={max_value} in '{label}'"
                logger.warning(err)
                self.validation_errors.append(err)
                results = False

        return results

    # ------------------------------------------------
    # Schema Validation
    # ------------------------------------------------

    def validate_schema(self, df: pd.DataFrame) -> bool:

        missing = set(self.required_columns) - set(df.columns)

        if missing:

            err = f"Missing required columns: {missing}"

            logger.error(err)

            self.validation_errors.append(err)

            return False

        return True

    # ------------------------------------------------
    # Null Validation
    # ------------------------------------------------

    def validate_nulls(self, df: pd.DataFrame) -> bool:

        ratios = df.isnull().mean()

        problematic = ratios[ratios > self.max_null_ratio]

        if not problematic.empty:

            err = f"High null ratios detected: {problematic.to_dict()}"

            logger.warning(err)

            self.validation_errors.append(err)

            return False

        return True

    # ------------------------------------------------
    # Duplicate Validation
    # ------------------------------------------------

    def validate_duplicates(self, df: pd.DataFrame) -> bool:

        if "text" not in df.columns:
            return True

        dup_count = df.duplicated(subset=["text"]).sum()

        dup_ratio = dup_count / len(df)

        if dup_ratio > self.max_dup_ratio:

            err = f"High duplicate ratio: {dup_ratio:.2%}"

            logger.warning(err)

            self.validation_errors.append(err)

            return False

        return True

    # ------------------------------------------------
    # Multi-task Label Validation
    # ------------------------------------------------

    def validate_labels(self, df: pd.DataFrame) -> bool:

        results = True

        for label in self.label_columns:

            if label not in df.columns:
                continue

            values = df[label].dropna()

            if values.empty:
                err = f"Label '{label}' has no non-null values"
                logger.warning(err)
                self.validation_errors.append(err)
                results = False
                continue

            if not self._validate_label_spec(label=label, values=values):
                results = False

            if values.nunique() < 2:

                err = f"Label '{label}' has <2 classes"

                logger.warning(err)

                self.validation_errors.append(err)

                results = False

                continue

            distribution = values.value_counts(normalize=True)

            if distribution.min() < self.min_class_ratio:

                err = f"Class imbalance in '{label}': {distribution.to_dict()}"

                logger.warning(err)

                self.validation_errors.append(err)

                results = False

        return results

    # ------------------------------------------------
    # Text Quality Validation
    # ------------------------------------------------

    def validate_text_quality(self, df: pd.DataFrame) -> bool:

        if "text" not in df.columns:
            return True

        lengths = df["text"].astype(str).str.len()

        short_ratio = (lengths < self.min_text_length).mean()

        if short_ratio > 0.1:

            err = f"Too many short texts: {short_ratio:.2%}"

            logger.warning(err)

            self.validation_errors.append(err)

            return False

        return True

    # ------------------------------------------------
    # Vocabulary Validation
    # ------------------------------------------------

    def validate_vocabulary(self, df: pd.DataFrame) -> bool:

        if "text" not in df.columns:
            return True

        words = " ".join(df["text"].astype(str)).split()

        vocab_size = len(set(words))

        if vocab_size < 100:

            err = f"Vocabulary too small: {vocab_size}"

            logger.warning(err)

            self.validation_errors.append(err)

            return False

        return True

    # ------------------------------------------------
    # Dataset Size Check
    # ------------------------------------------------

    def validate_dataset_size(self, df: pd.DataFrame):

        if len(df) < 100:

            logger.warning("Dataset extremely small (<100 samples)")

    # ------------------------------------------------
    # Dataset Summary
    # ------------------------------------------------

    def dataset_summary(self, df: pd.DataFrame) -> Dict[str, Any]:

        text_series = df["text"].astype(str) if "text" in df.columns else pd.Series()

        words = " ".join(text_series).split()

        summary = {

            "rows": len(df),
            "columns": list(df.columns),

            "avg_text_length": int(text_series.str.len().mean()) if not text_series.empty else 0,
            "median_text_length": int(text_series.str.len().median()) if not text_series.empty else 0,

            "vocab_size": len(set(words)),
        }

        label_stats = {}

        for label in self.label_columns:

            if label in df.columns:

                label_stats[label] = df[label].value_counts().to_dict()

        summary["label_distribution"] = label_stats

        return summary

    # ------------------------------------------------
    # Full Validation
    # ------------------------------------------------

    def validate(self, df: pd.DataFrame) -> Dict[str, Any]:

        if df.empty:
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

        self.validate_dataset_size(df)

        results["dataset_summary"] = self.dataset_summary(df)

        results["errors"] = self.validation_errors

        results["all_passed"] = all(results.values())

        return results


def validate_dataset(
    csv_path: str,
    label_columns: List[str] | None = None,
    label_specs: Dict[str, Dict[str, Any]] | None = None,
) -> Dict[str, Any]:

    df = pd.read_csv(csv_path)

    validator = DataValidator(
        label_columns=label_columns,
        label_specs=label_specs,
    )

    return validator.validate(df)
