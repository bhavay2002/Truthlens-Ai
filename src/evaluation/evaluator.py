"""
File: evaluator.py (FINAL - CONFIG DRIVEN + RESEARCH GRADE)
"""

from __future__ import annotations

import logging
from typing import Dict, Any, Iterable, Optional, Callable, List

import numpy as np
import torch

# Core metrics
from .metrics import compute_classification_metrics, compute_multilabel_metrics

# New calibration pipeline
from .calibration import compute_calibration

# Task config (CRITICAL)
from src.config.task_config import (
    get_task_type,
    get_threshold,
    use_auto_threshold,
)

# Explainability
from src.explainability.explanation_metrics import ExplanationMetrics
from src.explainability.explanation_consistency import ExplanationConsistency

# Feature importance
from src.features.importance.feature_ablation import FeatureAblation
from src.features.importance.permutation_importance import PermutationImportance
from src.features.importance.shap_importance import ShapImportance

logger = logging.getLogger(__name__)


# =========================================================
# DEVICE
# =========================================================
def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# =========================================================
# ACTIVATIONS
# =========================================================
def softmax(x):
    e = np.exp(x - np.max(x, axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


# =========================================================
# EVALUATOR
# =========================================================
class Evaluator:

    # =====================================================
    # VALIDATION
    # =====================================================
    @staticmethod
    def _validate_inputs(y_true, y_pred=None, y_proba=None):

        if y_true is None:
            raise ValueError("y_true cannot be None")

        y_true = np.asarray(y_true)

        if y_true.size == 0:
            raise ValueError("y_true cannot be empty")

        if y_pred is not None:
            y_pred = np.asarray(y_pred)
            if y_true.shape != y_pred.shape:
                raise ValueError(
                    f"Shape mismatch: y_true {y_true.shape} vs y_pred {y_pred.shape}"
                )

        if y_proba is not None:
            y_proba = np.asarray(y_proba)
            if y_proba.shape[0] != y_true.shape[0]:
                raise ValueError("y_proba mismatch with y_true")

    # =====================================================
    # MAIN ENTRYPOINT (CONFIG-DRIVEN)
    # =====================================================
    @staticmethod
    def evaluate(
        *,
        y_true: Iterable,
        y_pred: Optional[Iterable] = None,
        y_proba: Optional[Iterable] = None,
        model=None,
        X=None,
        task: str,
        batch_size: int = 32,
    ) -> Dict[str, Any]:

        # ---------------------------
        # TASK CONFIG
        # ---------------------------
        task_type = get_task_type(task)

        logger.info(
            f"[EVAL] task={task} | type={task_type} | samples={len(y_true)} | mode={'model' if model else 'direct'}"
        )

        # ---------------------------
        # MODEL MODE
        # ---------------------------
        logits = None

        if model is not None:
            if X is None:
                raise ValueError("X must be provided when model is used")

            logits = Evaluator._batched_predict(model, X, task, batch_size)
            y_pred, y_proba = Evaluator._postprocess(logits, task_type)

        # ---------------------------
        # VALIDATION
        # ---------------------------
        Evaluator._validate_inputs(y_true, y_pred, y_proba)

        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)

        # ---------------------------
        # METRICS
        # ---------------------------
        if task_type == "multilabel":
            metrics = compute_multilabel_metrics(
                y_true,
                y_pred,
                y_proba=y_proba,
                threshold=get_threshold(task),
                auto_threshold=use_auto_threshold(task),
            )
        else:
            metrics = compute_classification_metrics(
                y_true,
                y_pred,
                y_proba,
            )

        # ---------------------------
        # CALIBRATION (NEW PIPELINE)
        # ---------------------------
        if y_proba is not None:
            try:
                metrics["calibration"] = compute_calibration(
                    logits=logits,
                    y_true=y_true,
                    task_type=task_type,
                )
            except Exception as e:
                logger.warning(f"[CALIBRATION FAILED] {e}")

        return metrics

    # =====================================================
    # BATCHED INFERENCE
    # =====================================================
    @staticmethod
    def _batched_predict(model, X, task, batch_size):

        device = get_device()
        model.to(device)
        model.eval()

        outputs = []

        logger.info(f"[INFER] task={task} | batch_size={batch_size}")

        with torch.no_grad():
            for i in range(0, len(X), batch_size):

                batch = torch.tensor(X[i:i + batch_size]).to(device)

                out = model.predict(batch, task=task)

                if "logits" not in out:
                    raise ValueError("model.predict must return 'logits'")

                logits = out["logits"].detach().cpu().numpy()
                outputs.append(logits)

        return np.vstack(outputs)

    # =====================================================
    # POSTPROCESS
    # =====================================================
    @staticmethod
    def _postprocess(logits, task_type):

        if task_type == "multiclass":
            probs = softmax(logits)
            preds = np.argmax(probs, axis=1)

        elif task_type == "binary":
            probs = sigmoid(logits).reshape(-1)
            preds = (probs > 0.5).astype(int)

        elif task_type == "multilabel":
            probs = sigmoid(logits)
            preds = (probs > 0.5).astype(int)

        else:
            raise ValueError(f"Unknown task_type: {task_type}")

        return preds, probs

    # =====================================================
    # MULTITASK AGGREGATION
    # =====================================================
    @staticmethod
    def multitask(results: Dict[str, Dict[str, Any]]):

        summary = {}

        weights = {
            t: r.get("dataset_stats", {}).get("num_samples", 1)
            for t, r in results.items()
        }

        for metric in ["accuracy", "f1", "precision", "recall"]:

            vals, wts = [], []

            for t, r in results.items():
                val = r.get(metric)

                if val is not None:
                    vals.append(val)
                    wts.append(weights[t])

            if vals:
                arr = np.array(vals)

                summary[f"weighted_{metric}"] = float(
                    np.average(arr, weights=wts)
                )
                summary[f"std_{metric}"] = float(np.std(arr))
                summary[f"var_{metric}"] = float(np.var(arr))

        return summary

    # =====================================================
    # EXPLANATION METRICS
    # =====================================================
    @staticmethod
    def explanation_metrics(
        tokens: List[str],
        scores: List[float],
        predict_fn: Callable,
    ):
        evaluator = ExplanationMetrics()

        return evaluator.evaluate(
            tokens=tokens,
            scores=scores,
            predict_fn=predict_fn,
        )

    # =====================================================
    # EXPLANATION CONSISTENCY
    # =====================================================
    @staticmethod
    def explanation_consistency(**kwargs):
        consistency = ExplanationConsistency()
        return consistency.compute(**kwargs)

    # =====================================================
    # FEATURE IMPORTANCE (TASK-AWARE)
    # =====================================================
    @staticmethod
    def _predict_fn(model, task):

        def fn(X):
            device = get_device()

            if not torch.is_tensor(X):
                X = torch.tensor(X).to(device)

            out = model.predict(X, task=task)

            return out["logits"].detach().cpu().numpy()

        return fn

    @staticmethod
    def feature_importance_ablation(
        model,
        X,
        y,
        feature_names,
        task,
        metric=None,
        top_k=None,
    ):
        logger.info(f"[ABLATION] task={task}")

        ablator = FeatureAblation(metric=metric)

        scores = ablator.single_feature_ablation(
            X=X,
            y=y,
            feature_names=feature_names,
            predict_fn=Evaluator._predict_fn(model, task),
        )

        result = {
            "scores": scores,
            "ranked": ablator.rank_features(scores),
        }

        if top_k:
            result["top_k"] = ablator.top_k(scores, top_k)

        return result

    @staticmethod
    def feature_importance_permutation(
        model,
        X,
        y,
        feature_names,
        task,
        n_repeats=5,
        random_seed=42,
        metric=None,
        top_k=None,
    ):
        logger.info(f"[PERMUTATION] task={task}")

        perm = PermutationImportance(metric=metric)

        scores = perm.compute(
            X=X,
            y=y,
            feature_names=feature_names,
            predict_fn=Evaluator._predict_fn(model, task),
            n_repeats=n_repeats,
            random_seed=random_seed,
        )

        result = {
            "scores": scores,
            "ranked": perm.rank_features(scores),
        }

        if top_k:
            result["top_k"] = perm.top_k(scores, top_k)

        return result

    @staticmethod
    def feature_importance_shap(
        model,
        X,
        feature_names,
        task,
        top_k=None,
    ):
        logger.info(f"[SHAP] task={task}")

        shap_calc = ShapImportance(model=None)

        scores = shap_calc.compute_with_function(
            predict_fn=Evaluator._predict_fn(model, task),
            X=X,
            feature_names=feature_names,
        )

        result = {
            "scores": scores,
            "ranked": sorted(scores.items(), key=lambda x: x[1], reverse=True),
        }

        if top_k:
            result["top_k"] = result["ranked"][:top_k]

        return result