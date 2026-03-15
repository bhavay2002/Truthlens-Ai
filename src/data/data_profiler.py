"""
File: src/data/data_profiler.py

Purpose
-------
Automatic dataset profiling for NLP datasets.

Generates dataset quality reports including:
- dataset statistics
- label distribution
- vocabulary analysis
- duplicate ratio
- text length statistics

Outputs
-------
reports/
 ├── dataset_profile.json
 └── dataset_quality_report.md

Inputs
------
df : pandas.DataFrame
csv_path : str

Outputs
-------
profile statistics dictionary
saved reports

Dependencies
------------
pandas
json
logging
pathlib
collections
"""

from __future__ import annotations

import json
import logging
from collections import Counter
from pathlib import Path
from typing import Dict, Any

import pandas as pd

logger = logging.getLogger(__name__)


class DataProfiler:
    """
    Dataset profiling tool for NLP datasets.

    Parameters
    ----------
    df : pandas.DataFrame
        Dataset containing text and optional label column.

    report_dir : Path
        Directory to save profiling reports.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        text_column: str = "text",
        label_column: str = "label",
        report_dir: str | Path = "reports",
    ):

        self.df = df.copy()
        self.text_column = text_column
        self.label_column = label_column

        self.report_dir = Path(report_dir)
        self.report_dir.mkdir(parents=True, exist_ok=True)

        self.summary: Dict[str, Any] = {}

    # -------------------------------------------------
    # Basic Dataset Statistics
    # -------------------------------------------------

    def dataset_overview(self) -> None:

        self.summary["rows"] = len(self.df)
        self.summary["columns"] = len(self.df.columns)
        self.summary["column_names"] = list(self.df.columns)

        logger.info("Dataset rows: %s", len(self.df))

    # -------------------------------------------------
    # Missing Values
    # -------------------------------------------------

    def missing_values(self) -> None:

        missing = self.df.isnull().sum().to_dict()

        self.summary["missing_values"] = missing

    # -------------------------------------------------
    # Duplicate Detection
    # -------------------------------------------------

    def duplicate_analysis(self) -> None:

        dup_count = self.df.duplicated().sum()
        dup_ratio = (dup_count / len(self.df)) if len(self.df) > 0 else 0.0

        self.summary["duplicate_count"] = int(dup_count)
        self.summary["duplicate_ratio"] = float(dup_ratio)

    # -------------------------------------------------
    # Label Distribution
    # -------------------------------------------------

    def label_distribution(self) -> None:

        if self.label_column in self.df.columns:

            distribution = self.df[self.label_column].value_counts().to_dict()

            self.summary["label_distribution"] = distribution

    # -------------------------------------------------
    # Text Length Statistics
    # -------------------------------------------------

    def text_length_stats(self) -> None:

        if self.text_column not in self.df.columns:
            return

        text_lengths = self.df[self.text_column].astype(str).str.len()
        if text_lengths.empty:
            self.summary["avg_text_length"] = 0
            self.summary["median_text_length"] = 0
            self.summary["max_text_length"] = 0
            self.summary["min_text_length"] = 0
            return

        self.summary["avg_text_length"] = int(text_lengths.mean())
        self.summary["median_text_length"] = int(text_lengths.median())
        self.summary["max_text_length"] = int(text_lengths.max())
        self.summary["min_text_length"] = int(text_lengths.min())

    # -------------------------------------------------
    # Vocabulary Analysis
    # -------------------------------------------------

    def vocabulary_analysis(self) -> None:

        if self.text_column not in self.df.columns:
            return

        words = []

        for text in self.df[self.text_column].astype(str):

            words.extend(text.split())

        vocab = set(words)

        self.summary["vocab_size"] = len(vocab)

        if words:

            diversity = len(vocab) / len(words)

            self.summary["lexical_diversity"] = float(diversity)

    # -------------------------------------------------
    # Most Common Words
    # -------------------------------------------------

    def most_common_words(self, top_n: int = 20) -> None:

        if self.text_column not in self.df.columns:
            return

        words = []

        for text in self.df[self.text_column].astype(str):

            words.extend(text.lower().split())

        counts = Counter(words)

        self.summary["top_words"] = counts.most_common(top_n)

    # -------------------------------------------------
    # Save JSON Report
    # -------------------------------------------------

    def save_json(self) -> None:

        json_path = self.report_dir / "dataset_profile.json"

        with json_path.open("w", encoding="utf-8") as f:

            json.dump(self.summary, f, indent=2)

        logger.info("Dataset profile saved to %s", json_path)

    # -------------------------------------------------
    # Save Markdown Report
    # -------------------------------------------------

    def save_markdown(self) -> None:

        md_path = self.report_dir / "dataset_quality_report.md"

        lines = []

        lines.append("# Dataset Quality Report\n")

        for key, value in self.summary.items():

            lines.append(f"## {key}\n")
            lines.append(f"```\n{value}\n```\n")

        with md_path.open("w", encoding="utf-8") as f:

            f.write("\n".join(lines))

        logger.info("Dataset report saved to %s", md_path)

    # -------------------------------------------------
    # Run Full Profiling
    # -------------------------------------------------

    def profile(self) -> Dict[str, Any]:

        logger.info("Running dataset profiler")

        self.dataset_overview()
        self.missing_values()
        self.duplicate_analysis()
        self.label_distribution()

        self.text_length_stats()
        self.vocabulary_analysis()
        self.most_common_words()

        self.save_json()
        self.save_markdown()

        logger.info("Dataset profiling completed")

        return self.summary


# -------------------------------------------------
# Convenience Function
# -------------------------------------------------

def profile_dataset(csv_path: str) -> Dict[str, Any]:
    """
    Run dataset profiling from CSV file.
    """

    df = pd.read_csv(csv_path)

    profiler = DataProfiler(df)

    return profiler.profile()
