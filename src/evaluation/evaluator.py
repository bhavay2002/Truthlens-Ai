from __future__ import annotations

import logging
from typing import Dict, Any, Iterable, Optional, List

import numpy as np
import torch
from transformers import AutoTokenizer

from src.utils.device_utils import move_batch, autocast_context
from src.utils.metrics_utils import compute_task_metrics
from src.evaluation.calibration import compute_calibration
from src.config.task_config import get_task_type

#  NEW
from src.evaluation.prediction_collector import PredictionCollector
from src.evaluation.error_analysis import ErrorAnalyzer
from src.evaluation.threshold_optimizer import ThresholdOptimizer

logger = logging.getLogger(__name__)


# =========================================================
# TOKENIZATION
# =========================================================

def _tokenize(tokenizer, texts: List[str], max_length=512):
    return tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )


# =========================================================
# EVALUATOR
# =========================================================

class Evaluator:

    def __init__(self):
        #  NEW
        self.collector = PredictionCollector()
        self.error_analyzer = ErrorAnalyzer()
        self.threshold_optimizer = ThresholdOptimizer()

    # =====================================================
    # MODEL INFERENCE
    # =====================================================

    @staticmethod
    def _batched_predict(
        model,
        texts: List[str],
        task: str,
        tokenizer: AutoTokenizer,
        batch_size: int,
    ):

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        model.to(device)
        model.eval()

        outputs = []

        with torch.no_grad():
            for i in range(0, len(texts), batch_size):

                batch_texts = texts[i:i + batch_size]

                encoded = _tokenize(tokenizer, batch_texts)
                encoded = move_batch(encoded, device)

                with autocast_context():
                    out = model(
                        input_ids=encoded["input_ids"],
                        attention_mask=encoded["attention_mask"],
                        task=task,
                    )

                logits = out["logits"].detach().cpu().numpy()
                outputs.append(logits)

        return np.vstack(outputs)

    # =====================================================
    # POSTPROCESS
    # =====================================================

    @staticmethod
    def _postprocess(logits, task_type):

        logits = np.asarray(logits)
        logits_t = torch.tensor(logits)

        if task_type == "multiclass":
            probs = torch.softmax(logits_t, dim=-1).numpy()
            preds = np.argmax(probs, axis=1)

        elif task_type == "binary":
            probs = torch.sigmoid(logits_t).numpy().reshape(-1)
            preds = (probs >= 0.5).astype(int)

        elif task_type == "multilabel":
            probs = torch.sigmoid(logits_t).numpy()
            preds = (probs >= 0.5).astype(int)

        else:
            raise ValueError(f"Unknown task_type: {task_type}")

        return preds, probs

    # =====================================================
    # MAIN ENTRYPOINT (UPDATED )
    # =====================================================

    def evaluate(
        self,
        *,
        y_true: Iterable,
        task: str,
        y_pred: Optional[Iterable] = None,
        y_proba: Optional[Iterable] = None,
        model=None,
        texts: Optional[List[str]] = None,
        tokenizer: Optional[AutoTokenizer] = None,
        batch_size: int = 32,
        return_logits: bool = False,
    ) -> Dict[str, Any]:

        task_type = get_task_type(task)
        logits = None

        # ==================================================
        # MODEL MODE
        # ==================================================
        if model is not None:

            if texts is None or tokenizer is None:
                raise ValueError("Model mode requires texts + tokenizer")

            logits = self._batched_predict(
                model,
                texts,
                task,
                tokenizer,
                batch_size,
            )

            y_pred, y_proba = self._postprocess(logits, task_type)

        # ==================================================
        # VALIDATION
        # ==================================================
        y_true = np.asarray(y_true)

        if y_true.size == 0:
            raise ValueError("Empty labels")

        if y_pred is None:
            raise ValueError("y_pred must be provided if model is None")

        y_pred = np.asarray(y_pred)

        # ==================================================
        #  COLLECTION (NEW)
        # ==================================================
        collected = self.collector.collect(
            y_true=y_true,
            y_pred=y_pred,
            y_proba=y_proba,
            logits=logits,
        )

        # ==================================================
        # METRICS
        # ==================================================
        metric_input = y_proba if y_proba is not None else y_pred

        metrics = compute_task_metrics(
            logits=torch.tensor(metric_input),
            labels=torch.tensor(y_true),
            task_type=task_type,
            num_labels=None,
        )

        # ==================================================
        # CALIBRATION + RELIABILITY
        # ==================================================
        calibration = {}
        if logits is not None:
            try:
                calibration = compute_calibration(
                    logits=logits,
                    y_true=y_true,
                    task_type=task_type,
                )
            except Exception as e:
                logger.warning(f"Calibration failed: {e}")

        # ==================================================
        #  ERROR ANALYSIS (NEW)
        # ==================================================
        error_analysis = self.error_analyzer.analyze(collected)

        # ==================================================
        #  THRESHOLD OPTIMIZATION (NEW)
        # ==================================================
        thresholds = None
        if y_proba is not None:
            try:
                thresholds = self.threshold_optimizer.optimize(collected)
            except Exception as e:
                logger.warning(f"Threshold optimization failed: {e}")

        # ==================================================
        # DATASET STATS
        # ==================================================
        if y_true.ndim == 1:
            labels, counts = np.unique(y_true, return_counts=True)

            dataset_stats = {
                "num_samples": int(len(y_true)),
                "num_classes": int(len(labels)),
                "class_distribution": dict(
                    zip(labels.astype(str), counts.tolist())
                ),
            }

        else:
            dataset_stats = {
                "num_samples": int(y_true.shape[0]),
                "num_labels": int(y_true.shape[1]),
                "density": float(np.mean(y_true)),
            }

        # ==================================================
        # FINAL OUTPUT (UPDATED )
        # ==================================================
        result = {
            "task": task,
            "task_type": task_type,
            "metrics": metrics,
            "calibration": calibration,
            "error_analysis": error_analysis,
            "optimal_thresholds": thresholds,
            "dataset_stats": dataset_stats,
        }

        if return_logits and logits is not None:
            result["logits"] = logits

        return result

    # =====================================================
    # MULTITASK SUMMARY
    # =====================================================

    @staticmethod
    def multitask(results: Dict[str, Dict[str, Any]]):

        summary = {}

        weights = {
            t: r["dataset_stats"]["num_samples"]
            for t, r in results.items()
        }

        for metric in ["accuracy", "f1_macro", "f1_weighted"]:

            vals, wts = [], []

            for t, r in results.items():
                val = r["metrics"].get(metric)

                if val is not None:
                    vals.append(val)
                    wts.append(weights[t])

            if vals:
                summary[f"weighted_{metric}"] = float(
                    np.average(vals, weights=wts)
                )

        return summary