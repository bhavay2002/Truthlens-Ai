"""
File Name: evaluate_saved_model.py
Module: TruthLens AI - Model Evaluation Runner
Description:
    Production-grade evaluation script for evaluating saved TruthLens models.
    Supports multi-task evaluation, advanced diagnostics, calibration analysis,
    uncertainty estimation, task correlation analysis, MLflow experiment
    tracking, and PDF/JSON reporting.

Dependencies:
    logging
    typing
    pathlib
    json
    numpy
    pandas
    src.evaluation.evaluator
    src.evaluation.advanced_analysis
    src.evaluation.calibration
    src.evaluation.uncertainty
    src.evaluation.task_correlation
    src.evaluation.mlflow_tracker
    src.evaluation.report_writer
    src.evaluation.pdf_report

Inputs:
    preds: dictionary of predictions for each task
    labels: dictionary of ground truth labels
    pred_probs: probability outputs (optional)
    df: optional dataframe for advanced analysis

Outputs:
    structured evaluation report JSON and PDF
"""

from __future__ import annotations

import logging
from typing import Dict, Tuple, Any
from pathlib import Path
import json

import numpy as np
import pandas as pd

from src.evaluation.evaluator import Evaluator
from src.evaluation.advanced_analysis import (
    actor_graph_metrics,
    frame_coherence,
)
from src.evaluation.calibration import expected_calibration_error
from src.evaluation.uncertainty import uncertainty_statistics
from src.evaluation.task_correlation import compute_task_correlation
from src.evaluation.mlflow_tracker import start_experiment, log_metrics
from src.evaluation.report_writer import save_report
from src.evaluation.pdf_report import generate_pdf_report


logger = logging.getLogger(__name__)


SUPPORTED_TASKS = [
    "bias",
    "ideology",
    "propaganda",
    "frame",
    "emotion",
]


def _validate_inputs(
    preds: Dict[str, Any],
    labels: Dict[str, Any],
) -> None:

    if not isinstance(preds, dict):
        raise TypeError("preds must be a dictionary")

    if not isinstance(labels, dict):
        raise TypeError("labels must be a dictionary")

    for task in SUPPORTED_TASKS:

        if task not in preds:
            raise ValueError(f"Missing predictions for task: {task}")

        if task not in labels:
            raise ValueError(f"Missing labels for task: {task}")

        if len(preds[task]) != len(labels[task]):
            raise ValueError(
                f"Prediction/label length mismatch for task {task}"
            )


def _evaluate_core_tasks(
    preds: Dict[str, Any],
    labels: Dict[str, Any],
) -> Dict[str, Dict[str, Any]]:

    results: Dict[str, Dict[str, Any]] = {}

    logger.info("Evaluating classification tasks")

    results["bias"] = Evaluator.classification(
        labels["bias"],
        preds["bias"],
    )

    results["ideology"] = Evaluator.classification(
        labels["ideology"],
        preds["ideology"],
    )

    results["propaganda"] = Evaluator.classification(
        labels["propaganda"],
        preds["propaganda"],
    )

    results["frame"] = Evaluator.classification(
        labels["frame"],
        preds["frame"],
    )

    logger.info("Evaluating multilabel emotion task")

    results["emotion"] = Evaluator.multilabel(
        labels["emotion"],
        preds["emotion"],
    )

    return results


def _run_advanced_analysis(
    df: pd.DataFrame | None,
    preds: Dict[str, Any],
    labels: Dict[str, Any],
) -> Dict[str, Any]:

    diagnostics: Dict[str, Any] = {}

    try:

        if df is not None:
            diagnostics["actor_graph"] = actor_graph_metrics(df)

    except Exception as exc:
        logger.warning("Actor graph analysis failed: %s", exc)

    try:

        diagnostics["frame_coherence"] = frame_coherence(
            preds["frame"],
            labels["frame"],
        )

    except Exception as exc:
        logger.warning("Frame coherence computation failed: %s", exc)

    return diagnostics


def evaluate_tasks(
    preds: Dict[str, Any],
    labels: Dict[str, Any],
    df: pd.DataFrame | None = None,
) -> Tuple[Dict[str, Any], Dict[str, float]]:

    logger.info("Starting TruthLens multi-task evaluation")

    _validate_inputs(preds, labels)

    results = _evaluate_core_tasks(preds, labels)

    summary = Evaluator.multitask(results)

    diagnostics = _run_advanced_analysis(df, preds, labels)

    results["advanced_analysis"] = diagnostics

    logger.info("Evaluation complete")

    return results, summary


def evaluate_and_save(
    preds: Dict[str, Any],
    labels: Dict[str, Any],
    output_path: str | Path,
    pred_probs: np.ndarray | None = None,
    df: pd.DataFrame | None = None,
) -> Dict[str, Any]:

    results, summary = evaluate_tasks(preds, labels, df)

    task_corr = compute_task_correlation(preds)

    ece = None
    uncertainty = None

    try:

        if pred_probs is not None:
            ece = expected_calibration_error(
                labels["bias"],
                pred_probs[:, 1]
            )

            uncertainty = uncertainty_statistics(pred_probs)

    except Exception as exc:
        logger.warning("Calibration/uncertainty computation failed: %s", exc)

    report = {
        "tasks": results,
        "summary": summary,
        "task_correlation": task_corr.to_dict(),
        "ece": ece,
        "uncertainty": uncertainty,
    }

    save_report(report, output_path)

    pdf_path = Path(output_path).with_suffix(".pdf")

    generate_pdf_report(report, pdf_path)

    logger.info("Evaluation reports saved")

    return report


def load_predictions(pred_path: str | Path) -> Dict[str, Any]:

    path = Path(pred_path)

    if not path.exists():
        raise FileNotFoundError(f"Prediction file not found: {path}")

    with path.open("r", encoding="utf-8") as f:
        preds = json.load(f)

    return preds


def load_labels(label_path: str | Path) -> Dict[str, Any]:

    path = Path(label_path)

    if not path.exists():
        raise FileNotFoundError(f"Label file not found: {path}")

    with path.open("r", encoding="utf-8") as f:
        labels = json.load(f)

    return labels


def run_evaluation(
    pred_path: str | Path,
    label_path: str | Path,
    output_report: str | Path,
    pred_probs: np.ndarray | None = None,
    dataset_path: str | Path | None = None,
) -> Dict[str, Any]:

    logger.info("Running TruthLens evaluation pipeline")

    preds = load_predictions(pred_path)
    labels = load_labels(label_path)

    df = None

    if dataset_path is not None:
        dataset_path = Path(dataset_path)

        if dataset_path.exists():
            df = pd.read_csv(dataset_path)

    run = start_experiment()

    report = evaluate_and_save(
        preds=preds,
        labels=labels,
        output_path=output_report,
        pred_probs=pred_probs,
        df=df,
    )

    log_metrics(report["summary"])

    logger.info("MLflow experiment logged")

    return report