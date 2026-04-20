"""
File Name: experiment_utils.py
Module: src.utils
Description:
    Experiment tracking utilities for TruthLens AI.

    This module provides structured experiment tracking capabilities for
    machine learning experiments. It supports experiment ID generation,
    experiment metadata creation, runtime measurement, and persistent
    logging of experiment records.

    Designed to support reproducible ML research workflows and production
    experiment monitoring.

Author: TruthLens Engineering
Date: 2026-04-03
Dependencies:
    - Python 3.10+

Inputs:
    - Model metadata
    - Training parameters
    - Evaluation metrics

Outputs:
    - Structured experiment records
    - Persisted experiment logs (JSON)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Optional
from dataclasses import dataclass, asdict

from src.utils.json_utils import append_json
from src.utils.time_utils import timestamp


# ---------------------------------------------------------
# Logging
# ---------------------------------------------------------

logger = logging.getLogger(__name__)


# ---------------------------------------------------------
# Experiment Record Dataclass
# ---------------------------------------------------------


@dataclass(slots=True)
class ExperimentRecord:
    """
    Structured representation of an ML experiment.
    """

    experiment_id: str
    timestamp: str
    model: str
    dataset: Optional[str]
    parameters: Dict[str, Any]
    metrics: Dict[str, Any]
    runtime_seconds: Optional[float]


# ---------------------------------------------------------
# Experiment ID
# ---------------------------------------------------------


def generate_experiment_id(prefix: str = "exp") -> str:
    """
    Generate a unique experiment identifier.

    Parameters
    ----------
    prefix : str
        Prefix for experiment identifiers.

    Returns
    -------
    str
        Unique experiment ID.
    """

    return f"{prefix}_{timestamp()}"


# ---------------------------------------------------------
# Experiment Record Builder
# ---------------------------------------------------------


def create_experiment_record(
    model_name: str,
    parameters: Dict[str, Any],
    metrics: Dict[str, Any],
    dataset: Optional[str] = None,
    runtime: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Create structured experiment record.

    Parameters
    ----------
    model_name : str
        Model name used in experiment.
    parameters : Dict[str, Any]
        Training hyperparameters.
    metrics : Dict[str, Any]
        Evaluation metrics.
    dataset : Optional[str]
        Dataset name.
    runtime : Optional[float]
        Runtime duration in seconds.

    Returns
    -------
    Dict[str, Any]
        Serialized experiment record.
    """

    ts = timestamp()
    record = ExperimentRecord(
        experiment_id=f"exp_{ts}",
        timestamp=ts,
        model=model_name,
        dataset=dataset,
        parameters=parameters,
        metrics=metrics,
        runtime_seconds=runtime,
    )

    return asdict(record)


# ---------------------------------------------------------
# Save Experiment
# ---------------------------------------------------------

_REQUIRED_EXPERIMENT_KEYS = {
    "experiment_id",
    "timestamp",
    "model",
    "dataset",
    "parameters",
    "metrics",
    "runtime_seconds",
}


def log_experiment(
    record: Dict[str, Any],
    output_path: str | Path = "reports/experiments.json",
) -> Path:
    """
    Persist experiment record.

    Parameters
    ----------
    record : Dict[str, Any]
        Experiment record.
    output_path : str | Path
        File where experiment history is stored.

    Returns
    -------
    Path
        Path to experiment log file.
    """

    try:
        missing = _REQUIRED_EXPERIMENT_KEYS - set(record.keys())
        if missing:
            raise ValueError(
                f"Experiment record missing required keys: {sorted(missing)}"
            )
        path = append_json(record, output_path)

        logger.info(
            "Experiment logged successfully: %s",
            record.get("experiment_id"),
        )

        return path

    except (TypeError, ValueError):
        raise
    except OSError as exc:
        logger.exception("Failed to write experiment log file")
        raise RuntimeError("Experiment logging failed due to file I/O error") from exc
    except RuntimeError as exc:
        logger.exception("Failed to log experiment")
        raise RuntimeError("Experiment logging failed") from exc