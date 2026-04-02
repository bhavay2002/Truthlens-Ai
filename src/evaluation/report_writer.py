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

import json
import logging
from pathlib import Path
from typing import Dict, Any


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
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with output_path.open("w", encoding="utf-8") as f:
            json.dump(
                report,
                f,
                indent=4,
                ensure_ascii=False
            )

        logger.info("Evaluation report saved to %s", output_path)

    except Exception as exc:
        logger.exception("Failed to save evaluation report")
        raise RuntimeError("Report writing failed") from exc

    return output_path