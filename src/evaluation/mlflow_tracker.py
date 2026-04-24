"""
File: mlflow_tracker.py (FINAL - INDUSTRY + RESEARCH GRADE)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Any, Optional

try:
    import mlflow
except ImportError:
    mlflow = None

logger = logging.getLogger(__name__)


# =========================================================
# SAFETY
# =========================================================
def _ensure_mlflow():
    if mlflow is None:
        raise RuntimeError("MLflow not installed")


# =========================================================
# UTIL: FLATTEN DICT
# =========================================================
def flatten_dict(d: Dict[str, Any], parent_key="", sep="."):
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


# =========================================================
# MAIN RUN CONTEXT (UPGRADED)
# =========================================================
class MLflowRun:

    def __init__(
        self,
        experiment_name: str = "truthlens",
        run_name: Optional[str] = None,
        tags: Optional[Dict[str, str]] = None,
    ):
        self.experiment_name = experiment_name
        self.run_name = run_name
        self.tags = tags or {}

    def __enter__(self):
        _ensure_mlflow()

        mlflow.set_experiment(self.experiment_name)

        self.run = mlflow.start_run(run_name=self.run_name)

        if self.tags:
            mlflow.set_tags(self.tags)

        logger.info(
            f"[MLFLOW] Run started | experiment={self.experiment_name} | run_name={self.run_name}"
        )

        return self

    def __exit__(self, exc_type, exc, tb):
        status = "FAILED" if exc else "FINISHED"
        mlflow.end_run(status=status)

        logger.info(f"[MLFLOW] Run ended with status={status}")


# =========================================================
# NESTED RUNS (PER TASK)
# =========================================================
class NestedRun:

    def __init__(self, name: str, tags: Optional[Dict[str, str]] = None):
        self.name = name
        self.tags = tags or {}

    def __enter__(self):
        _ensure_mlflow()

        self.run = mlflow.start_run(run_name=self.name, nested=True)

        if self.tags:
            mlflow.set_tags(self.tags)

        logger.info(f"[MLFLOW] Nested run started | {self.name}")

        return self

    def __exit__(self, exc_type, exc, tb):
        status = "FAILED" if exc else "FINISHED"
        mlflow.end_run(status=status)

        logger.info(f"[MLFLOW] Nested run ended | {self.name} | {status}")


# =========================================================
# METRIC LOGGING (MULTI-TASK SAFE)
# =========================================================
def log_metrics(
    metrics: Dict[str, Any],
    step: Optional[int] = None,
    prefix: str = "",
):
    _ensure_mlflow()

    flat = flatten_dict(metrics)

    for key, value in flat.items():

        if not isinstance(value, (int, float)):
            continue

        name = f"{prefix}{key}" if prefix else key

        try:
            mlflow.log_metric(name, float(value), step=step)
        except Exception:
            logger.warning(f"[MLFLOW] Failed metric: {name}")


# =========================================================
# PARAM LOGGING (FLATTENED)
# =========================================================
def log_params(params: Dict[str, Any], prefix=""):
    _ensure_mlflow()

    flat = flatten_dict(params)

    for key, value in flat.items():
        name = f"{prefix}{key}" if prefix else key

        try:
            mlflow.log_param(name, value)
        except Exception:
            logger.warning(f"[MLFLOW] Failed param: {name}")


# =========================================================
# DATASET VERSION LOGGING (NEW 🔥)
# =========================================================
def log_dataset_info(
    dataset_name: str,
    version: Optional[str] = None,
    size: Optional[int] = None,
    hash: Optional[str] = None,
):
    """
    Log dataset metadata for reproducibility.
    """

    _ensure_mlflow()

    try:
        mlflow.log_param("dataset.name", dataset_name)

        if version:
            mlflow.log_param("dataset.version", version)

        if size:
            mlflow.log_param("dataset.size", size)

        if hash:
            mlflow.log_param("dataset.hash", hash)

        logger.info(f"[MLFLOW] Dataset logged: {dataset_name}")

    except Exception as e:
        logger.warning(f"[MLFLOW] Dataset logging failed: {e}")


# =========================================================
# ARTIFACT LOGGING
# =========================================================
def log_artifact(path: str | Path, artifact_path: str = "artifacts"):
    _ensure_mlflow()

    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(path)

    try:
        mlflow.log_artifact(str(path), artifact_path=artifact_path)
        logger.info(f"[MLFLOW] Artifact logged: {path}")
    except Exception as e:
        logger.error(f"[MLFLOW] Artifact failed: {e}")


# =========================================================
# MODEL LOGGING
# =========================================================
def log_model(model, name="model"):
    _ensure_mlflow()

    try:
        import mlflow.pytorch
        mlflow.pytorch.log_model(model, name)
        logger.info("[MLFLOW] Model logged")
    except Exception as e:
        logger.warning(f"[MLFLOW] Model logging failed: {e}")


# =========================================================
# SYSTEM INFO
# =========================================================
def log_system_info():
    _ensure_mlflow()

    import platform
    import sys

    mlflow.log_param("system.python_version", sys.version)
    mlflow.log_param("system.platform", platform.platform())


# =========================================================
# EXTRA: TAG HELPERS
# =========================================================
def set_tags(tags: Dict[str, str]):
    _ensure_mlflow()

    try:
        mlflow.set_tags(tags)
    except Exception:
        logger.warning("[MLFLOW] Failed to set tags")