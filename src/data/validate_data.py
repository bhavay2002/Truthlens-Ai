"""
Data validation utilities
Ensures dataset quality before ML training
"""

import pandas as pd
import logging
from typing import List, Dict, Any
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataValidator:
    """
    Validates dataset quality and structure before training
    """

    def __init__(
        self,
        required_columns: List[str] = None,
        max_null_ratio: float = 0.1,
        max_dup_ratio: float = 0.2,
        min_class_ratio: float = 0.2,
        min_text_length: int = 10
    ):
        self.required_columns = required_columns or ["text", "label"]
        self.max_null_ratio = max_null_ratio
        self.max_dup_ratio = max_dup_ratio
        self.min_class_ratio = min_class_ratio
        self.min_text_length = min_text_length
        self.validation_errors = []

    # ------------------------------------------------
    # Schema Validation
    # ------------------------------------------------

    def validate_schema(self, df: pd.DataFrame) -> bool:
        """Check if required columns exist"""
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

    def validate_nulls(self, df: pd.DataFrame) -> bool:
        """Check for excessive null values"""
        null_ratios = df[self.required_columns].isnull().mean()

        problematic = null_ratios[null_ratios > self.max_null_ratio]

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
        """Check duplicate texts"""
        dup_count = df.duplicated(subset=["text"]).sum()
        dup_ratio = dup_count / len(df)

        if dup_ratio > self.max_dup_ratio:
            warning = f"High duplicate ratio: {dup_ratio:.2%} ({dup_count} duplicates)"
            logger.warning(warning)
            self.validation_errors.append(warning)
            return False

        return True

    # ------------------------------------------------
    # Label Validation
    # ------------------------------------------------

    def validate_labels(self, df: pd.DataFrame) -> bool:
        """Validate label distribution"""
        if "label" not in df.columns:
            return True

        label_counts = df["label"].value_counts(normalize=True)

        min_ratio = label_counts.min()

        if min_ratio < self.min_class_ratio:
            warning = f"Class imbalance detected: {label_counts.to_dict()}"
            logger.warning(warning)
            self.validation_errors.append(warning)
            return False

        return True

    # ------------------------------------------------
    # Text Quality
    # ------------------------------------------------

    def validate_text_quality(self, df: pd.DataFrame) -> bool:
        """Check text length quality"""

        if "text" not in df.columns:
            return True

        df["text"] = df["text"].astype(str)

        short_texts = (df["text"].str.len() < self.min_text_length).sum()
        short_ratio = short_texts / len(df)

        if short_ratio > 0.1:
            warning = (
                f"Too many short texts: {short_ratio:.2%} "
                f"({short_texts} texts < {self.min_text_length} chars)"
            )
            logger.warning(warning)
            self.validation_errors.append(warning)
            return False

        return True

    # ------------------------------------------------
    # Vocabulary Sanity Check
    # ------------------------------------------------

    def validate_vocabulary(self, df: pd.DataFrame) -> bool:
        """Check vocabulary size"""

        if "text" not in df.columns:
            return True

        words = " ".join(df["text"].astype(str)).split()
        vocab_size = len(set(words))

        if vocab_size < 50:
            warning = f"Very small vocabulary detected: {vocab_size} words"
            logger.warning(warning)
            self.validation_errors.append(warning)
            return False

        return True

    # ------------------------------------------------
    # Dataset Summary
    # ------------------------------------------------

    def dataset_summary(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Generate dataset statistics"""

        summary = {
            "rows": len(df),
            "columns": len(df.columns),
            "avg_text_length": int(df["text"].astype(str).str.len().mean()),
            "median_text_length": int(df["text"].astype(str).str.len().median()),
            "vocab_size": len(
                set(" ".join(df["text"].astype(str)).split())
            ),
        }

        if "label" in df.columns:
            summary["label_distribution"] = df["label"].value_counts().to_dict()

        return summary

    # ------------------------------------------------
    # Run All Validations
    # ------------------------------------------------

    def validate(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Run all validations"""

        if len(df) == 0:
            raise ValueError("Dataset is empty!")

        logger.info("Running dataset validation...")

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

        all_passed = all(results.values())
        results["all_passed"] = all_passed
        results["errors"] = self.validation_errors

        if all_passed:
            logger.info("✓ Dataset validation passed")
        else:
            logger.warning("⚠ Dataset validation issues detected")
            for err in self.validation_errors:
                logger.warning(f" - {err}")

        logger.info(f"Dataset summary: {results['dataset_summary']}")

        return results


# ------------------------------------------------
# Convenience Function
# ------------------------------------------------

def validate_dataset(csv_path: str) -> Dict[str, Any]:
    """Validate dataset from CSV file"""

    df = pd.read_csv(csv_path)

    validator = DataValidator()

    return validator.validate(df)