from __future__ import annotations

import logging
from typing import Any, Dict, Iterable, List, Optional

import numpy as np
import torch
from transformers import AutoTokenizer

from src.config.task_config import get_task_type
from src.evaluation.calibration import compute_calibration
from src.evaluation.error_analysis import ErrorAnalyzer
from src.evaluation.metrics_engine import compute_metrics_from_preds
from src.evaluation.prediction_collector import PredictionCollector
from src.evaluation.threshold_optimizer import ThresholdOptimizer
from src.utils.device_utils import autocast_context, move_batch

logger = logging.getLogger(__name__)


# =========================================================
# TOKENIZATION
# =========================================================

def _tokenize(tokenizer, texts: List[str], max_length: int = 512):
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
    ) -> np.ndarray:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        model.eval()

        outputs: List[np.ndarray] = []
        with torch.no_grad():
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i: i + batch_size]
                encoded = _tokenize(tokenizer, batch_texts)
                encoded = move_batch(encoded, device)

                with autocast_context():
                    out = model(
                        input_ids=encoded["input_ids"],
                        attention_mask=encoded["attention_mask"],
                        task=task,
                    )

                outputs.append(out["logits"].detach().cpu().numpy())

        return np.vstack(outputs)

    @staticmethod
    def _postprocess(logits: np.ndarray, task_type: str):
        arr = np.asarray(logits, dtype=float)
        logits_t = torch.tensor(arr, dtype=torch.float32)

        if task_type == "multiclass":
            probs = torch.softmax(logits_t, dim=-1).numpy()
            preds = np.argmax(probs, axis=1).astype(int)
        elif task_type == "binary":
            if logits_t.ndim == 2 and logits_t.shape[-1] == 2:
                probs_full = torch.softmax(logits_t, dim=-1).numpy()
                probs = probs_full[:, 1]
            else:
                probs = torch.sigmoid(logits_t).numpy().reshape(-1)
            preds = (probs >= 0.5).astype(int)
        elif task_type == "multilabel":
            probs = torch.sigmoid(logits_t).numpy()
            preds = (probs >= 0.5).astype(int)
        else:
            raise ValueError(f"Unknown task_type: {task_type}")

        return preds, probs

    # =====================================================
    # MAIN ENTRYPOINT
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
        logits: Optional[np.ndarray] = None

        if model is not None:
            if texts is None or tokenizer is None:
                raise ValueError("model mode requires texts + tokenizer")
            logits = self._batched_predict(model, texts, task, tokenizer, batch_size)
            y_pred, y_proba = self._postprocess(logits, task_type)

        y_true_arr = np.asarray(y_true)
        if y_true_arr.size == 0:
            raise ValueError("y_true cannot be empty")

        if y_pred is None:
            raise ValueError("y_pred must be provided if model is None")

        y_pred_arr = np.asarray(y_pred)
        y_proba_arr = np.asarray(y_proba, dtype=float) if y_proba is not None else None

        collected = self.collector.collect(
            y_true=y_true_arr,
            y_pred=y_pred_arr,
            y_proba=y_proba_arr,
            logits=logits,
            task=task,
            task_type=task_type,
        )

        metrics = compute_metrics_from_preds(
            y_true=y_true_arr,
            y_pred=y_pred_arr,
            task_type=task_type,
            y_proba=y_proba_arr,
        )

        calibration: Dict[str, Any] = {}
        if logits is not None:
            try:
                calibration = compute_calibration(
                    logits=logits, y_true=y_true_arr, task_type=task_type
                )
            except Exception as exc:
                logger.warning("Calibration failed: %s", exc)

        try:
            error_analysis = self.error_analyzer.analyze(collected)
        except Exception as exc:
            logger.warning("Error analysis failed: %s", exc)
            error_analysis = {}

        thresholds = None
        if y_proba_arr is not None:
            try:
                thresholds = self.threshold_optimizer.optimize(collected)
            except Exception as exc:
                logger.warning("Threshold optimization failed: %s", exc)

        if y_true_arr.ndim == 1:
            labels, counts = np.unique(y_true_arr, return_counts=True)
            class_counts = {str(int(label)): int(count) for label, count in zip(labels, counts)}
            dataset_stats = {
                "num_samples": int(len(y_true_arr)),
                "num_classes": int(len(labels)),
                "class_counts": class_counts,
                "class_distribution": class_counts,
            }
        else:
            dataset_stats = {
                "num_samples": int(y_true_arr.shape[0]),
                "num_labels": int(y_true_arr.shape[1]),
                "density": float(np.mean(y_true_arr)),
            }

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
    def multitask(results: Dict[str, Dict[str, Any]]) -> Dict[str, float]:
        if not results:
            return {}

        weights: Dict[str, float] = {}
        for task, result in results.items():
            stats = result.get("dataset_stats") or {}
            weights[task] = float(stats.get("num_samples", 1) or 1)

        summary: Dict[str, float] = {}
        for metric in ("accuracy", "f1_macro", "f1_weighted"):
            vals: List[float] = []
            wts: List[float] = []
            for task, result in results.items():
                value = (result.get("metrics") or {}).get(metric)
                if isinstance(value, (int, float)):
                    vals.append(float(value))
                    wts.append(weights[task])
            if vals:
                summary[f"weighted_{metric}"] = float(np.average(vals, weights=wts))
        return summary

    # =====================================================
    # FEATURE IMPORTANCE — STATIC TEST-FACING API
    # =====================================================

    @staticmethod
    def feature_importance_ablation(
        *,
        model: Any,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: List[str],
        scoring: str = "accuracy",
    ) -> Dict[str, float]:
        """Compute per-feature ablation importance.

        For each feature, replace its column with the column mean, score the
        model, and record the drop in score relative to the baseline. The
        result is a ``{feature_name: importance}`` mapping where larger values
        indicate features the model relies on more.
        """
        from sklearn.metrics import accuracy_score, f1_score

        X = np.asarray(X)
        y = np.asarray(y)

        if X.ndim != 2:
            raise ValueError("X must be 2D")
        if X.shape[1] != len(feature_names):
            raise ValueError("feature_names length must match X.shape[1]")

        if scoring == "accuracy":
            score_fn = lambda yt, yp: accuracy_score(yt, yp)
        elif scoring == "f1":
            score_fn = lambda yt, yp: f1_score(yt, yp, average="macro", zero_division=0)
        else:
            raise ValueError(f"Unsupported scoring: {scoring}")

        baseline = score_fn(y, model.predict(X))
        importances: Dict[str, float] = {}

        for idx, name in enumerate(feature_names):
            X_ablated = X.copy()
            X_ablated[:, idx] = float(np.mean(X[:, idx]))
            try:
                preds = model.predict(X_ablated)
                ablated_score = score_fn(y, preds)
            except Exception as exc:
                logger.warning("Ablation failed for %s: %s", name, exc)
                ablated_score = baseline

            importances[name] = float(baseline - ablated_score)

        return importances

    @staticmethod
    def feature_importance_shap(
        *,
        model: Any,
        X: np.ndarray,
        feature_names: List[str],
        max_samples: int = 100,
    ) -> Dict[str, float]:
        """Compute SHAP-style importance for tabular ``model``.

        ``max_samples`` controls how many rows the explainer sees and must be
        positive. Falls back to a permutation-based importance signal when the
        ``shap`` library is unavailable so the public contract stays stable.
        """
        if max_samples <= 0:
            raise ValueError("max_samples must be > 0")

        X = np.asarray(X)
        if X.ndim != 2:
            raise ValueError("X must be 2D")
        if X.shape[1] != len(feature_names):
            raise ValueError("feature_names length must match X.shape[1]")

        sample = X[: min(max_samples, X.shape[0])]

        try:
            import shap  # type: ignore

            explainer = shap.Explainer(model.predict, sample)
            shap_values = explainer(sample).values
            mean_abs = np.mean(np.abs(shap_values), axis=0)
        except Exception as exc:
            logger.debug("Falling back to permutation importance: %s", exc)
            baseline_pred = np.asarray(model.predict(sample))
            mean_abs = np.zeros(sample.shape[1], dtype=float)
            rng = np.random.default_rng(0)
            for idx in range(sample.shape[1]):
                perturbed = sample.copy()
                perturbed[:, idx] = rng.permutation(perturbed[:, idx])
                try:
                    preds = np.asarray(model.predict(perturbed))
                except Exception:
                    continue
                mean_abs[idx] = float(np.mean(preds != baseline_pred))

        return {
            name: float(mean_abs[idx]) for idx, name in enumerate(feature_names)
        }


__all__ = ["Evaluator"]
