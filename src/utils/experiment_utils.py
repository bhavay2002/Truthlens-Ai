"""
File: experiment_utils.py

Purpose
-------
Experiment tracking utilities for TruthLens AI.

This module records experiment metadata and results so
multiple training runs can be compared and reproduced.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict

from src.utils.json_utils import append_json
from src.utils.time_utils import timestamp


# ---------------------------------------------------------
# Logging
# ---------------------------------------------------------

logger = logging.getLogger(__name__)


# ---------------------------------------------------------
# Experiment ID
# ---------------------------------------------------------

def generate_experiment_id(prefix: str = "exp") -> str:
    """
    Generate unique experiment identifier.
    """

    return f"{prefix}_{timestamp()}"


# ---------------------------------------------------------
# Build Experiment Record
# ---------------------------------------------------------

def create_experiment_record(
    model_name: str,
    parameters: Dict[str, Any],
    metrics: Dict[str, Any],
    dataset: str | None = None,
    runtime: float | None = None,
) -> Dict[str, Any]:
    """
    Create structured experiment record.
    """

    record = {
        "experiment_id": generate_experiment_id(),
        "timestamp": timestamp(),
        "model": model_name,
        "dataset": dataset,
        "parameters": parameters,
        "metrics": metrics,
        "runtime_seconds": runtime,
    }

    return record


# ---------------------------------------------------------
# Save Experiment
# ---------------------------------------------------------

def log_experiment(
    record: Dict[str, Any],
    output_path: str | Path = "reports/experiments.json",
) -> Path:
    """
    Save experiment record.
    """

    try:

        path = append_json(record, output_path)

        logger.info("Experiment logged: %s", record["experiment_id"])

        return path

    except Exception:

        logger.exception("Failed to log experiment")

        raise