"""
File Name: mlflow_tracker.py
Module: TruthLens AI - MLflow Tracking
Description:
    MLflow tracking utilities used by TruthLens AI for experiment management.
    Provides safe wrappers around MLflow APIs for starting experiments,
    logging parameters, metrics, and artifacts, and ensuring reproducible
    experiment metadata. Designed to integrate with training and evaluation
    pipelines.
Dependencies:
    mlflow
    logging
    pathlib
    typing
Inputs:
    name: experiment name
    metrics: dictionary of metric values
    params: dictionary of parameter values
    path: artifact path
Outputs:
    Active MLflow run and logged experiment metadata
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Any, Optional

import mlflow


logger = logging.getLogger(__name__)


def start_experiment(name: str = "truthlens_evaluation") -> mlflow.ActiveRun:
    """
    Start or attach to an MLflow experiment.
    """

    if not isinstance(name, str) or not name.strip():
        raise ValueError("Experiment name must be a non-empty string.")

    try:
        mlflow.set_experiment(name)
        run = mlflow.start_run()
        logger.info("Started MLflow run under experiment '%s'", name)
        return run
    except Exception as exc:
        logger.exception("Failed to start MLflow experiment")
        raise RuntimeError("MLflow experiment start failed") from exc


def log_metrics(metrics: Dict[str, Any]) -> None:
    """
    Log numeric metrics to MLflow.
    """

    if not isinstance(metrics, dict):
        raise TypeError("metrics must be a dictionary.")

    for key, value in metrics.items():

        if isinstance(value, (int, float)):

            try:
                mlflow.log_metric(key, float(value))
            except Exception:
                logger.warning("Failed to log metric: %s", key)

        else:
            logger.debug("Skipping non-numeric metric: %s", key)


def log_params(params: Dict[str, Any]) -> None:
    """
    Log parameters to MLflow.
    """

    if not isinstance(params, dict):
        raise TypeError("params must be a dictionary.")

    for key, value in params.items():

        try:
            mlflow.log_param(key, value)
        except Exception:
            logger.warning("Failed to log parameter: %s", key)


def log_artifact(path: str | Path) -> Path:
    """
    Log artifact file to MLflow.
    """

    artifact_path = Path(path)

    if not artifact_path.exists():
        raise FileNotFoundError(f"Artifact not found: {artifact_path}")

    try:
        mlflow.log_artifact(str(artifact_path))
        logger.info("Logged artifact to MLflow: %s", artifact_path)
    except Exception as exc:
        logger.exception("Failed to log artifact")
        raise RuntimeError("Artifact logging failed") from exc

    return artifact_path


def end_run(status: Optional[str] = None) -> None:
    """
    End the active MLflow run.
    """

    try:
        mlflow.end_run(status=status)
        logger.info("MLflow run ended")
    except Exception:
        logger.warning("Failed to properly close MLflow run")