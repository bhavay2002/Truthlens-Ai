from __future__ import annotations

import logging
from typing import Dict, Any, Optional
from pathlib import Path
import json

import numpy as np
import pandas as pd

from src.evaluation.evaluate_model import evaluate
from src.evaluation.calibration import compute_calibration
from src.evaluation.uncertainty import uncertainty_statistics
from src.evaluation.task_correlation import compute_task_correlation
from src.evaluation.report_writer import save_report
from src.config.task_config import TASK_CONFIG

logger = logging.getLogger(__name__)


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
# CORE TASK EVALUATION
# =========================================================

def evaluate_tasks(
    preds: Dict[str, Any],
    labels: Dict[str, Any],
    pred_probs: Optional[Dict[str, Any]] = None,
):
    results = {}

    for task, cfg in TASK_CONFIG.items():
        logger.info(f"[EVAL] Task: {task}")

        res = evaluate(
            y_true=labels[task],
            y_pred=preds[task],
            y_proba=(pred_probs or {}).get(task),
            task=task,
        )

        results[task] = res

    return results


# =========================================================
# CALIBRATION (STRICT: LOGITS ONLY)
# =========================================================

def compute_all_calibration(
    logits: Dict[str, Any],
    labels: Dict[str, Any],
):
    calibration_results = {}

    for task, cfg in TASK_CONFIG.items():

        if task not in logits:
            logger.warning(f"[CALIBRATION] Missing logits for {task}")
            continue

        try:
            calibration_results[task] = compute_calibration(
                logits=np.asarray(logits[task]),
                y_true=np.asarray(labels[task]),
                task_type=cfg["type"],
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
    *,
    pred_probs: Optional[Dict[str, Any]] = None,
    logits: Optional[Dict[str, Any]] = None,
    df: Optional[pd.DataFrame] = None,
) -> Dict[str, Any]:

    validate_inputs(preds, labels)

    # -----------------------------------------------------
    # TASK EVALUATION
    # -----------------------------------------------------
    task_results = evaluate_tasks(preds, labels, pred_probs)

    # -----------------------------------------------------
    # SUMMARY
    # -----------------------------------------------------
    summary = {
        task: task_results[task]["metrics"]
        for task in task_results
    }

    # -----------------------------------------------------
    # ADVANCED ANALYSIS
    # -----------------------------------------------------
    advanced = {}

    if df is not None:
        try:
            from src.evaluation.advanced_analysis import actor_graph_metrics
            advanced["actor_graph"] = actor_graph_metrics(df)
        except Exception as e:
            logger.warning(f"Advanced analysis failed: {e}")

    # -----------------------------------------------------
    # CALIBRATION (LOGITS ONLY)
    # -----------------------------------------------------
    if logits is None:
        logger.warning(
            "[CALIBRATION] Skipped: logits not provided (REQUIRED for proper calibration)"
        )
        calibration = {}
    else:
        calibration = compute_all_calibration(logits, labels)

    # -----------------------------------------------------
    # UNCERTAINTY (TASK-AWARE)
    # -----------------------------------------------------
    uncertainty = {}

    if pred_probs:
        for task, probs in pred_probs.items():
            try:
                uncertainty[task] = uncertainty_statistics(
                    probs,
                    task=task,
                )
            except Exception as e:
                logger.warning(f"Uncertainty failed for {task}: {e}")

    # -----------------------------------------------------
    # TASK CORRELATION (PRIORITY FIXED)
    # -----------------------------------------------------
    try:
        corr_input = (
            pred_probs if pred_probs is not None
            else logits if logits is not None
            else preds
        )

        task_corr = compute_task_correlation(corr_input)

        if hasattr(task_corr, "to_dict"):
            task_corr = task_corr.to_dict()

    except Exception as e:
        logger.warning(f"Task correlation failed: {e}")
        task_corr = {}

    # -----------------------------------------------------
    # FINAL REPORT
    # -----------------------------------------------------
    report = {
        "tasks": task_results,
        "summary": summary,
        "advanced_analysis": advanced,
        "calibration": calibration,
        "uncertainty": uncertainty,
        "task_correlation": task_corr,
    }

    # -----------------------------------------------------
    # SAVE REPORT
    # -----------------------------------------------------
    save_report(report, output_path)

    # -----------------------------------------------------
    # PDF REPORT
    # -----------------------------------------------------
    try:
        from src.evaluation.pdf_report import generate_pdf_report
        generate_pdf_report(report, Path(output_path).with_suffix(".pdf"))
    except Exception as e:
        logger.warning(f"PDF generation failed: {e}")

    logger.info("[EVALUATION] Complete")

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
    *,
    pred_probs=None,
    logits=None,
    dataset_path=None,
):

    preds = load_json(pred_path)
    labels = load_json(label_path)

    df = None
    if dataset_path and Path(dataset_path).exists():
        df = pd.read_csv(dataset_path)

    report = evaluate_and_save(
        preds=preds,
        labels=labels,
        output_path=output_report,
        pred_probs=pred_probs,
        logits=logits,
        df=df,
    )

    return report