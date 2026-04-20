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
from typing import Dict, Tuple, Any, Optional
from pathlib import Path
import json

import numpy as np
import pandas as pd

from src.evaluation.evaluator import Evaluator
from src.evaluation.calibration import expected_calibration_error
from src.evaluation.uncertainty import uncertainty_statistics
from src.evaluation.task_correlation import compute_task_correlation
from src.evaluation.report_writer import save_report


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
    pred_probs_by_task: Optional[Dict[str, Any]] = None,
) -> Dict[str, Dict[str, Any]]:

    results: Dict[str, Dict[str, Any]] = {}

    logger.info("Evaluating classification tasks")

    results["bias"] = Evaluator.classification(
        labels["bias"],
        preds["bias"],
        y_proba=(pred_probs_by_task or {}).get("bias"),
    )

    results["ideology"] = Evaluator.classification(
        labels["ideology"],
        preds["ideology"],
        y_proba=(pred_probs_by_task or {}).get("ideology"),
    )

    results["propaganda"] = Evaluator.classification(
        labels["propaganda"],
        preds["propaganda"],
        y_proba=(pred_probs_by_task or {}).get("propaganda"),
    )

    results["frame"] = Evaluator.classification(
        labels["frame"],
        preds["frame"],
        y_proba=(pred_probs_by_task or {}).get("frame"),
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
        from src.evaluation.advanced_analysis import (
            actor_graph_metrics,
            frame_coherence,
        )
    except ImportError as exc:
        logger.warning("Advanced analysis dependencies unavailable: %s", exc)
        return diagnostics

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
    pred_probs_by_task: Optional[Dict[str, Any]] = None,
) -> Tuple[Dict[str, Any], Dict[str, float]]:

    logger.info("Starting TruthLens multi-task evaluation")

    _validate_inputs(preds, labels)

    results = _evaluate_core_tasks(
        preds,
        labels,
        pred_probs_by_task=pred_probs_by_task,
    )

    summary = Evaluator.multitask(results)

    diagnostics = _run_advanced_analysis(df, preds, labels)

    results["advanced_analysis"] = diagnostics

    logger.info("Evaluation complete")

    return results, summary


def evaluate_and_save(
    preds: Dict[str, Any],
    labels: Dict[str, Any],
    output_path: str | Path,
    pred_probs: np.ndarray | Dict[str, Any] | None = None,
    df: pd.DataFrame | None = None,
) -> Dict[str, Any]:
    pred_probs_by_task: Dict[str, Any] = {}
    bias_probs: np.ndarray | None = None

    if isinstance(pred_probs, dict):
        pred_probs_by_task = pred_probs
        maybe_bias = pred_probs.get("bias")
        if maybe_bias is not None:
            maybe_bias_arr = np.asarray(maybe_bias)
            if maybe_bias_arr.ndim == 2 and maybe_bias_arr.shape[1] >= 2:
                bias_probs = maybe_bias_arr
    elif pred_probs is not None:
        pred_probs_arr = np.asarray(pred_probs)
        if pred_probs_arr.ndim == 2 and pred_probs_arr.shape[1] >= 2:
            bias_probs = pred_probs_arr
            pred_probs_by_task["bias"] = pred_probs_arr

    results, summary = evaluate_tasks(
        preds,
        labels,
        df,
        pred_probs_by_task=pred_probs_by_task,
    )

    try:
        task_corr = compute_task_correlation(preds)
        task_corr_dict = task_corr.to_dict()
    except Exception as exc:
        logger.warning("Task correlation failed: %s", exc)
        task_corr_dict = {}

    ece = None
    uncertainty = None

    try:

        if bias_probs is not None:
            ece = expected_calibration_error(
                labels["bias"],
                bias_probs[:, 1]
            )

            uncertainty = uncertainty_statistics(bias_probs)

    except Exception as exc:
        logger.warning("Calibration/uncertainty computation failed: %s", exc)

    report = {
        "tasks": results,
        "summary": summary,
        "task_correlation": task_corr_dict,
        "ece": ece,
        "uncertainty": uncertainty,
    }

    save_report(report, output_path)

    pdf_path = Path(output_path).with_suffix(".pdf")

    try:
        from src.evaluation.pdf_report import generate_pdf_report
    except ImportError as exc:
        logger.warning("PDF generation skipped (dependency missing): %s", exc)
    else:
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
    pred_probs: np.ndarray | Dict[str, Any] | None = None,
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

    tracker_available = True
    run_started = False
    try:
        from src.evaluation.mlflow_tracker import start_experiment, log_metrics, end_run
    except ImportError as exc:
        logger.warning("MLflow tracking disabled (dependency missing): %s", exc)
        tracker_available = False
    else:
        start_experiment()
        run_started = True

    try:
        report = evaluate_and_save(
            preds=preds,
            labels=labels,
            output_path=output_report,
            pred_probs=pred_probs,
            df=df,
        )

        if tracker_available and run_started:
            log_metrics(report["summary"])
            logger.info("MLflow experiment logged")
    finally:
        if tracker_available and run_started:
            end_run()

    return report
