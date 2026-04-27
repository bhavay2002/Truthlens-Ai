"""End-to-end evaluation pipeline that orchestrates collection → metrics →
calibration → uncertainty → error analysis → correlation → reporting."""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import numpy as np

from src.config.task_config import TASK_CONFIG
from src.evaluation.calibration import compute_calibration, fit_calibration
from src.evaluation.error_analysis import error_analysis
from src.evaluation.evaluate_model import evaluate
from src.evaluation.pdf_report import generate_pdf_report
from src.evaluation.prediction_collector import collect_all_tasks
from src.evaluation.report_writer import save_report
from src.evaluation.task_correlation import compute_task_correlation
from src.evaluation.threshold_optimizer import optimize_thresholds
from src.evaluation.uncertainty import uncertainty_statistics

try:
    from src.evaluation.mlflow_tracker import (
        log_evaluation_report,
        log_task_metrics,
    )
except Exception:  # pragma: no cover
    log_task_metrics = None
    log_evaluation_report = None

try:
    from src.inference.prediction_service import PredictionService
except Exception:  # pragma: no cover - optional dep at import time
    PredictionService = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)


# =========================================================
# PREDICTION SERVICE PATH (BATCHED)
# =========================================================

def _collect_via_prediction_service(
    prediction_service,
    texts: List[str],
    tasks: List[str],
    *,
    batch_size: int = 32,
) -> Dict[str, Dict[str, Any]]:
    buffers: Dict[str, Dict[str, list]] = {
        task: {"probabilities": [], "predictions": [], "logits": []}
        for task in tasks
    }

    predict_batch = getattr(prediction_service, "predict_batch", None)

    if callable(predict_batch):
        for i in range(0, len(texts), batch_size):
            batch_results = predict_batch(texts[i: i + batch_size])
            for result in batch_results:
                for task in tasks:
                    task_out = result["tasks"][task]
                    buffers[task]["probabilities"].append(task_out["probabilities"])
                    buffers[task]["predictions"].append(task_out["predictions"])
                    buffers[task]["logits"].append(task_out.get("logits"))
    else:
        for text in texts:
            result = prediction_service.predict(text)
            for task in tasks:
                task_out = result["tasks"][task]
                buffers[task]["probabilities"].append(task_out["probabilities"])
                buffers[task]["predictions"].append(task_out["predictions"])
                buffers[task]["logits"].append(task_out.get("logits"))

    out: Dict[str, Dict[str, Any]] = {}
    for task in tasks:
        record: Dict[str, Any] = {}
        for key, values in buffers[task].items():
            if any(v is None for v in values):
                # logits may be None when the service doesn't expose them.
                if key == "logits" and all(v is None for v in values):
                    continue
            try:
                record[key] = np.asarray(values)
            except ValueError:
                record[key] = values
        out[task] = record
    return out


# =========================================================
# MAIN PIPELINE
# =========================================================

def run_evaluation_pipeline(
    *,
    model=None,
    tokenizer=None,
    texts: List[str],
    labels: Dict[str, Any],
    tasks: Optional[List[str]] = None,
    output_path: Optional[str] = None,
    prediction_service=None,
    enable_calibration: bool = True,
    enable_threshold_opt: bool = True,
    enable_uncertainty: bool = True,
    enable_error_analysis: bool = True,
    enable_correlation: bool = True,
    val_logits: Optional[Dict[str, np.ndarray]] = None,
    val_labels: Optional[Dict[str, np.ndarray]] = None,
) -> Dict[str, Any]:
    """Run the full evaluation pipeline.

    ``val_logits`` / ``val_labels`` enable a fit-on-val / apply-on-test
    calibration split; when omitted the calibrator falls back to fitting on the
    same data it scores (with a warning).
    """
    tasks = tasks or list(TASK_CONFIG.keys())

    logger.info("[PIPELINE] Collecting predictions for %d tasks", len(tasks))
    if prediction_service is not None:
        predictions = _collect_via_prediction_service(prediction_service, texts, tasks)
    else:
        predictions = collect_all_tasks(
            model=model,
            texts=texts,
            tokenizer=tokenizer,
            tasks=tasks,
        )

    report: Dict[str, Any] = {"tasks": {}}
    all_probs: Dict[str, np.ndarray] = {}
    all_logits: Dict[str, np.ndarray] = {}
    all_confidence: Dict[str, np.ndarray] = {}
    all_uncertainty: Dict[str, float] = {}

    fitted_temperatures: Dict[str, float] = {}
    if val_logits and val_labels:
        for task in tasks:
            if task not in val_logits or task not in val_labels:
                continue
            try:
                t = fit_calibration(
                    val_logits=val_logits[task],
                    val_y_true=val_labels[task],
                    task_type=TASK_CONFIG[task]["type"],
                )
                if t is not None:
                    fitted_temperatures[task] = t
            except Exception as exc:
                logger.warning("Temperature fit failed for %s: %s", task, exc)

    for task in tasks:
        logger.info("[PIPELINE] task=%s", task)
        task_preds = predictions.get(task, {})

        logits = np.asarray(task_preds.get("logits")) if task_preds.get("logits") is not None else None
        probs = np.asarray(task_preds.get("probabilities", task_preds.get("y_proba")))
        preds = np.asarray(task_preds.get("predictions", task_preds.get("y_pred")))
        y_true = np.asarray(labels[task])

        eval_result = evaluate(
            y_true=y_true,
            y_pred=preds,
            y_proba=probs,
            task=task,
        )
        report["tasks"][task] = eval_result

        all_probs[task] = probs
        if logits is not None:
            all_logits[task] = logits

        if enable_threshold_opt:
            try:
                report.setdefault("optimal_thresholds", {})[task] = optimize_thresholds(
                    y_true=y_true, probs=probs, task=task
                )
            except Exception as exc:
                logger.warning("Threshold optimization failed for %s: %s", task, exc)

        if enable_calibration and logits is not None:
            try:
                cal = compute_calibration(
                    logits=logits,
                    y_true=y_true,
                    task_type=TASK_CONFIG[task]["type"],
                    temperature=fitted_temperatures.get(task),
                )
                report.setdefault("calibration", {})[task] = cal
                if "confidence" in cal:
                    all_confidence[task] = np.asarray(cal["confidence"])
            except Exception as exc:
                logger.warning("Calibration failed for %s: %s", task, exc)

        if enable_uncertainty:
            try:
                unc = uncertainty_statistics(
                    np.asarray(probs), task=task, logits=logits
                )
                report.setdefault("uncertainty", {})[task] = unc
                mean_entropy = unc.get("mean_entropy")
                if mean_entropy is not None:
                    all_uncertainty[task] = float(mean_entropy)
            except Exception as exc:
                logger.warning("Uncertainty failed for %s: %s", task, exc)

        if enable_error_analysis:
            try:
                report.setdefault("error_analysis", {})[task] = error_analysis(
                    y_true=y_true,
                    y_pred=preds,
                    probs=probs,
                    texts=texts,
                    task=task,
                )
            except Exception as exc:
                logger.warning("Error analysis failed for %s: %s", task, exc)

        if log_task_metrics is not None:
            try:
                log_task_metrics(task, eval_result.get("metrics", {}))
            except Exception:
                pass

    if enable_correlation:
        try:
            corr = compute_task_correlation(all_probs)
            report["task_correlation"] = corr.to_dict()
        except Exception as exc:
            logger.warning("Correlation failed: %s", exc)

    summary: Dict[str, float] = {}
    for task, data in report["tasks"].items():
        for k, v in (data.get("metrics") or {}).items():
            if isinstance(v, (int, float)):
                summary[f"{task}_{k}"] = float(v)
    report["summary"] = summary

    if output_path:
        save_report(report, output_path)
        try:
            generate_pdf_report(report, str(output_path).replace(".json", ".pdf"))
        except Exception as exc:
            logger.warning("PDF generation failed: %s", exc)

    if log_evaluation_report is not None:
        try:
            log_evaluation_report(report)
        except Exception:
            pass

    logger.info("[PIPELINE] Evaluation complete")
    return report
