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
from typing import Dict, Any, List

import pandas as pd

logger = logging.getLogger(__name__)


class DataProfiler:
    """
    Dataset profiling tool for NLP datasets.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        text_column: str = "text",
        label_columns: List[str] | None = None,
        report_dir: str | Path = "reports",
    ):

        self.df = df.copy()
        self.text_column = text_column
        self.label_columns = label_columns or []

        self.report_dir = Path(report_dir)
        self.report_dir.mkdir(parents=True, exist_ok=True)

        self.summary: Dict[str, Any] = {}

    # -------------------------------------------------
    # Dataset Overview
    # -------------------------------------------------

    def dataset_overview(self):

        self.summary["rows"] = len(self.df)
        self.summary["columns"] = len(self.df.columns)
        self.summary["column_names"] = list(self.df.columns)

        logger.info("Dataset rows: %d", len(self.df))

    # -------------------------------------------------
    # Missing Values
    # -------------------------------------------------

    def missing_values(self):

        missing = self.df.isnull().sum().to_dict()

        self.summary["missing_values"] = missing

    # -------------------------------------------------
    # Duplicate Detection
    # -------------------------------------------------

    def duplicate_analysis(self):

        dup_count = self.df.duplicated().sum()

        dup_ratio = dup_count / len(self.df) if len(self.df) else 0

        self.summary["duplicate_count"] = int(dup_count)
        self.summary["duplicate_ratio"] = float(dup_ratio)

    # -------------------------------------------------
    # Multi-Task Label Distribution
    # -------------------------------------------------

    def label_distribution(self):

        label_stats = {}

        for col in self.label_columns:

            if col in self.df.columns:

                label_stats[col] = (
                    self.df[col]
                    .value_counts(dropna=True)
                    .to_dict()
                )

        self.summary["label_distribution"] = label_stats

    # -------------------------------------------------
    # Text Length Statistics
    # -------------------------------------------------

    def text_length_stats(self):

        if self.text_column not in self.df.columns:
            return

        lengths = self.df[self.text_column].astype(str).str.len()

        if lengths.empty:
            return

        self.summary["text_length_stats"] = {
            "avg": int(lengths.mean()),
            "median": int(lengths.median()),
            "max": int(lengths.max()),
            "min": int(lengths.min()),
        }

    # -------------------------------------------------
    # Token Statistics
    # -------------------------------------------------

    def token_statistics(self):

        if self.text_column not in self.df.columns:
            return

        token_lengths = self.df[self.text_column].astype(str).apply(
            lambda x: len(x.split())
        )

        self.summary["token_stats"] = {
            "avg_tokens": float(token_lengths.mean()),
            "median_tokens": float(token_lengths.median()),
            "max_tokens": int(token_lengths.max()),
            "min_tokens": int(token_lengths.min()),
        }

    # -------------------------------------------------
    # Vocabulary Analysis
    # -------------------------------------------------

    def vocabulary_analysis(self):

        if self.text_column not in self.df.columns:
            return

        words = []

        for text in self.df[self.text_column].astype(str):

            words.extend(text.split())

        vocab = set(words)

        total_words = len(words)

        self.summary["vocabulary"] = {
            "vocab_size": len(vocab),
            "total_tokens": total_words,
            "lexical_diversity": (
                len(vocab) / total_words if total_words else 0
            ),
        }

    # -------------------------------------------------
    # Most Common Words
    # -------------------------------------------------

    def most_common_words(self, top_n: int = 20):

        if self.text_column not in self.df.columns:
            return

        words = []

        for text in self.df[self.text_column].astype(str):

            words.extend(text.lower().split())

        counts = Counter(words)

        self.summary["top_words"] = counts.most_common(top_n)

    # -------------------------------------------------
    # Dataset Quality Metrics
    # -------------------------------------------------

    def dataset_quality_metrics(self):

        if self.text_column not in self.df.columns:
            return

        lengths = self.df[self.text_column].astype(str).str.len()

        short_ratio = (lengths < 20).mean()

        self.summary["quality_metrics"] = {
            "short_text_ratio": float(short_ratio),
        }

    # -------------------------------------------------
    # Save JSON Report
    # -------------------------------------------------

    def save_json(self):

        path = self.report_dir / "dataset_profile.json"

        with path.open("w", encoding="utf-8") as f:
            json.dump(self.summary, f, indent=2)

        logger.info("Dataset profile saved: %s", path)

    # -------------------------------------------------
    # Save Markdown Report
    # -------------------------------------------------

    def save_markdown(self):

        path = self.report_dir / "dataset_quality_report.md"

        lines = ["# Dataset Quality Report\n"]

        for key, value in self.summary.items():

            lines.append(f"## {key}\n")
            lines.append("```")
            lines.append(str(value))
            lines.append("```\n")

        with path.open("w", encoding="utf-8") as f:
            f.write("\n".join(lines))

        logger.info("Dataset report saved: %s", path)

    # -------------------------------------------------
    # Run Profiling
    # -------------------------------------------------

    def profile(self) -> Dict[str, Any]:

        logger.info("Running dataset profiling")

        self.dataset_overview()
        self.missing_values()
        self.duplicate_analysis()
        self.label_distribution()

        self.text_length_stats()
        self.token_statistics()

        self.vocabulary_analysis()
        self.most_common_words()

        self.dataset_quality_metrics()

        self.save_json()
        self.save_markdown()

        logger.info("Dataset profiling completed")

        return self.summary


# -------------------------------------------------
# Convenience Function
# -------------------------------------------------

def profile_dataset(
    csv_path: str,
    text_column: str = "text",
    label_columns: List[str] | None = None,
) -> Dict[str, Any]:

    df = pd.read_csv(csv_path)

    profiler = DataProfiler(
        df,
        text_column=text_column,
        label_columns=label_columns,
    )

    return profiler.profile()