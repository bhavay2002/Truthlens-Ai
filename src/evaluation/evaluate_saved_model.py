#File: evaluate_saved_model.py 

from __future__ import annotations

import logging
from typing import Dict, Any, Optional
from pathlib import Path
import json

import numpy as np
import pandas as pd

from src.evaluation.evaluator import Evaluator
from src.evaluation.calibration import compute_calibration
from src.evaluation.uncertainty import uncertainty_statistics
from src.evaluation.task_correlation import compute_task_correlation
from src.evaluation.report_writer import save_report

logger = logging.getLogger(__name__)


# =========================================================
# TASK CONFIG (CRITICAL)
# =========================================================
TASK_CONFIG = {
    "bias": {"type": "multiclass"},
    "ideology": {"type": "multiclass"},
    "propaganda": {"type": "binary"},
    "frame": {"type": "multiclass"},
    "emotion": {"type": "multilabel"},
}


# =========================================================
# VALIDATION
# =========================================================
def validate_inputs(preds, labels):
    for task in TASK_CONFIG:
        if task not in preds or task not in labels:
            raise ValueError(f"Missing task: {task}")

        if len(preds[task]) != len(labels[task]):
            raise ValueError(f"Mismatch in {task}")


# =========================================================
# CORE EVALUATION LOOP (DYNAMIC)
# =========================================================
def evaluate_tasks(
    preds: Dict[str, Any],
    labels: Dict[str, Any],
    pred_probs: Optional[Dict[str, Any]] = None,
):
    results = {}

    for task, cfg in TASK_CONFIG.items():
        task_type = cfg["type"]

        logger.info(f"Evaluating task: {task}")

        results[task] = Evaluator.evaluate(
            y_true=labels[task],
            y_pred=preds[task],
            y_proba=(pred_probs or {}).get(task),
            task=task,
            task_type=task_type,
        )

    summary = Evaluator.multitask(results)

    return results, summary


# =========================================================
# CALIBRATION (PER TASK)
# =========================================================
def compute_all_calibration(
    preds,
    labels,
    pred_probs,
):
    calibration_results = {}

    if pred_probs is None:
        return calibration_results

    for task, cfg in TASK_CONFIG.items():
        probs = pred_probs.get(task)
        if probs is None:
            continue

        try:
            calibration_results[task] = compute_calibration(
                model=None,
                X=None,
                y=labels[task],
                task=task,
                task_type=cfg["type"],
                from_logits=False,
            )
        except Exception as e:
            logger.warning(f"Calibration failed for {task}: {e}")

    return calibration_results


# =========================================================
# MAIN PIPELINE
# =========================================================
def evaluate_and_save(
    preds: Dict[str, Any],
    labels: Dict[str, Any],
    output_path: str | Path,
    pred_probs: Optional[Dict[str, Any]] = None,
    df: Optional[pd.DataFrame] = None,
) -> Dict[str, Any]:

    validate_inputs(preds, labels)

    results, summary = evaluate_tasks(preds, labels, pred_probs)

    # -----------------------------
    # Advanced Analysis
    # -----------------------------
    diagnostics = {}

    if df is not None:
        try:
            from src.evaluation.advanced_analysis import actor_graph_metrics
            diagnostics["actor_graph"] = actor_graph_metrics(df)
        except Exception as e:
            logger.warning(f"Advanced analysis failed: {e}")

    results["advanced_analysis"] = diagnostics

    # -----------------------------
    # Calibration
    # -----------------------------
    calibration = compute_all_calibration(preds, labels, pred_probs)

    # -----------------------------
    # Uncertainty
    # -----------------------------
    uncertainty = {}

    if pred_probs:
        for task, probs in pred_probs.items():
            try:
                uncertainty[task] = uncertainty_statistics(probs)
            except Exception as e:
                logger.warning(f"Uncertainty failed for {task}: {e}")

    # -----------------------------
    # Task Correlation
    # -----------------------------
    try:
        task_corr = compute_task_correlation(preds).to_dict()
    except Exception as e:
        logger.warning(f"Task correlation failed: {e}")
        task_corr = {}

    # -----------------------------
    # FINAL REPORT
    # -----------------------------
    report = {
        "tasks": results,
        "summary": summary,
        "calibration": calibration,
        "uncertainty": uncertainty,
        "task_correlation": task_corr,
    }

    save_report(report, output_path)

    # -----------------------------
    # PDF REPORT
    # -----------------------------
    try:
        from src.evaluation.pdf_report import generate_pdf_report
        generate_pdf_report(report, Path(output_path).with_suffix(".pdf"))
    except Exception as e:
        logger.warning(f"PDF generation failed: {e}")

    logger.info("Evaluation pipeline complete")

    return report


# =========================================================
# LOADERS
# =========================================================
def load_json(path):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open("r") as f:
        return json.load(f)


# =========================================================
# ENTRYPOINT
# =========================================================
def run_evaluation(
    pred_path,
    label_path,
    output_report,
    pred_probs=None,
    dataset_path=None,
):
    preds = load_json(pred_path)
    labels = load_json(label_path)

    df = None
    if dataset_path and Path(dataset_path).exists():
        df = pd.read_csv(dataset_path)

    report = evaluate_and_save(
        preds,
        labels,
        output_report,
        pred_probs,
        df,
    )

    return report