"""
File Name: report_writer.py
Module: TruthLens AI - Evaluation Reports
Description:
    Utilities for writing structured evaluation reports produced by the
    TruthLens evaluation pipeline. Supports JSON serialization, automatic
    directory creation, validation of report structure, and safe file writes.
    Designed for compatibility with dashboards, experiment tracking systems,
    and research artifacts.
Dependencies:
    json
    logging
    pathlib
    typing
Inputs:
    report: Dictionary containing evaluation results
    path: Destination file path for the report
Outputs:
    Persisted JSON evaluation report
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Any

from src.utils import create_folder, save_json
from src.visualization.visualize import plot_feature_importance


logger = logging.getLogger(__name__)


def _validate_report(report: Dict[str, Any]) -> None:
    """
    Validate report structure before saving.
    """

    if not isinstance(report, dict):
        raise TypeError("Report must be a dictionary.")

    if "tasks" not in report:
        logger.warning("Report missing 'tasks' section.")

    if "summary" not in report:
        logger.warning("Report missing 'summary' section.")


def save_report(
    report: Dict[str, Any],
    path: str | Path
) -> Path:
    """
    Save evaluation report to disk in JSON format.
    """

    _validate_report(report)

    output_path = Path(path)

    try:
        create_folder(output_path.parent)
        save_json(report, output_path, indent=4)

        logger.info("Evaluation report saved to %s", output_path)

        summary = report.get("summary")
        if isinstance(summary, dict):
            numeric_summary = {
                k: float(v)
                for k, v in summary.items()
                if isinstance(v, (int, float))
            }
            if numeric_summary:
                figure_path = output_path.parent / "evaluation_summary_metrics.png"
                plot_feature_importance(
                    features=list(numeric_summary.keys()),
                    scores=list(numeric_summary.values()),
                    top_k=min(20, len(numeric_summary)),
                    save_path=figure_path,
                )

    except Exception as exc:
        logger.exception("Failed to save evaluation report")
        raise RuntimeError("Report writing failed") from exc

    return output_path