from __future__ import annotations

import logging
from typing import Dict, Any, List, Optional

import numpy as np

from src.evaluation.prediction_collector import collect_all_tasks
from src.evaluation.evaluate_model import evaluate
from src.evaluation.calibration import compute_calibration
from src.evaluation.uncertainty import uncertainty_statistics
from src.evaluation.task_correlation import compute_task_correlation
from src.evaluation.threshold_optimizer import optimize_thresholds
from src.evaluation.error_analysis import error_analysis
from src.evaluation.report_writer import save_report
from src.evaluation.pdf_report import generate_pdf_report
from src.inference.prediction_service import PredictionService

from src.config.task_config import TASK_CONFIG

try:
    from src.tracking.mlflow_tracker import (
        log_task_metrics,
        log_evaluation_report,
    )
except Exception:
    log_task_metrics = None
    log_evaluation_report = None

logger = logging.getLogger(__name__)


# =========================================================
# MAIN PIPELINE (UPGRADED)
# =========================================================

def run_evaluation_pipeline(
    *,
    model=None,
    tokenizer=None,
    texts: List[str],
    labels: Dict[str, Any],
    tasks: Optional[List[str]] = None,
    output_path: Optional[str] = None,
    prediction_service: Optional[PredictionService] = None,
    enable_calibration: bool = True,
    enable_threshold_opt: bool = True,
    enable_uncertainty: bool = True,
    enable_error_analysis: bool = True,
    enable_correlation: bool = True,
) -> Dict[str, Any]:

    tasks = tasks or list(TASK_CONFIG.keys())

    # =====================================================
    # STEP 1: PREDICTIONS ( SERVICE-AWARE)
    # =====================================================
    logger.info("[PIPELINE] Collecting predictions")

    if prediction_service:
        #  unified inference
        predictions = {
            task: {
                "probabilities": [],
                "predictions": [],
                "logits": [],
            }
            for task in tasks
        }

        for text in texts:
            result = prediction_service.predict(text)

            for task in tasks:
                task_out = result["tasks"][task]

                predictions[task]["probabilities"].append(task_out["probabilities"])
                predictions[task]["predictions"].append(task_out["predictions"])
                predictions[task]["logits"].append(task_out.get("logits"))

        # convert to numpy
        for task in tasks:
            for k in predictions[task]:
                predictions[task][k] = np.asarray(predictions[task][k])

    else:
        # fallback
        predictions = collect_all_tasks(
            model=model,
            texts=texts,
            tokenizer=tokenizer,
            tasks=tasks,
        )

    report: Dict[str, Any] = {
        "tasks": {},
    }

    all_probs = {}
    all_logits = {}
    all_confidence = {}
    all_uncertainty = {}

    # =====================================================
    # STEP 2: TASK-WISE EVALUATION
    # =====================================================
    for task in tasks:

        logger.info(f"[PIPELINE] Evaluating task: {task}")

        task_preds = predictions[task]

        logits = task_preds["logits"]
        probs = task_preds["probabilities"]
        preds = task_preds["predictions"]

        y_true = np.asarray(labels[task])

        # -------------------------
        # EVALUATION
        # -------------------------
        eval_result = evaluate(
            y_true=y_true,
            y_pred=preds,
            y_proba=probs,
            task=task,
        )

        report["tasks"][task] = eval_result

        # store
        all_probs[task] = probs
        all_logits[task] = logits

        # -------------------------
        # THRESHOLDS
        # -------------------------
        if enable_threshold_opt:
            try:
                th = optimize_thresholds(y_true, probs, task=task)
                report.setdefault("optimal_thresholds", {})[task] = th
            except Exception as e:
                logger.warning(f"Threshold optimization failed: {e}")

        # -------------------------
        # CALIBRATION (includes reliability)
        # -------------------------
        if enable_calibration:
            try:
                cal = compute_calibration(
                    logits=logits,
                    y_true=y_true,
                    task_type=TASK_CONFIG[task]["type"],
                )
                report.setdefault("calibration", {})[task] = cal

                #  extract confidence
                if "confidence" in cal:
                    all_confidence[task] = np.asarray(cal["confidence"])

            except Exception as e:
                logger.warning(f"Calibration failed: {e}")

        # -------------------------
        # UNCERTAINTY
        # -------------------------
        if enable_uncertainty:
            try:
                unc = uncertainty_statistics(
                    probs,
                    task=task,
                    logits=logits,
                )
                report.setdefault("uncertainty", {})[task] = unc

                all_uncertainty[task] = unc.get("mean_entropy")

            except Exception as e:
                logger.warning(f"Uncertainty failed: {e}")

        # -------------------------
        # ERROR ANALYSIS
        # -------------------------
        if enable_error_analysis:
            try:
                err = error_analysis(
                    y_true,
                    preds,
                    probs=probs,
                    texts=texts,
                    task=task,
                )
                report.setdefault("error_analysis", {})[task] = err
            except Exception as e:
                logger.warning(f"Error analysis failed: {e}")

        # -------------------------
        # LOGGING
        # -------------------------
        if log_task_metrics:
            try:
                log_task_metrics(task, eval_result["metrics"])
            except Exception:
                pass

    # =====================================================
    # STEP 3: CORRELATION ( UPGRADED)
    # =====================================================
    if enable_correlation:
        try:
            corr = compute_task_correlation(
                all_probs,
                confidence=np.mean(list(all_confidence.values()), axis=0)
                if all_confidence else None,
                uncertainty=np.mean(list(all_uncertainty.values()))
                if all_uncertainty else None,
            )

            report["task_correlation"] = corr.to_dict()

        except Exception as e:
            logger.warning(f"Correlation failed: {e}")

    # =====================================================
    # STEP 4: SUMMARY
    # =====================================================
    summary = {}

    for task, data in report["tasks"].items():
        for k, v in data.get("metrics", {}).items():
            if isinstance(v, (int, float)):
                summary[f"{task}_{k}"] = v

    report["summary"] = summary

    # =====================================================
    # STEP 5: SAVE
    # =====================================================
    if output_path:
        save_report(report, output_path)

        try:
            generate_pdf_report(report, output_path.replace(".json", ".pdf"))
        except Exception:
            logger.warning("PDF generation failed")

    # =====================================================
    # STEP 6: LOG FULL REPORT
    # =====================================================
    if log_evaluation_report:
        try:
            log_evaluation_report(report)
        except Exception:
            pass

    logger.info("[PIPELINE] Evaluation complete")

    return report