"""
File: postprocessing.py
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, Any, Optional

import numpy as np

logger = logging.getLogger(__name__)

EPS = 1e-12


# =========================================================
# CONFIG
# =========================================================

@dataclass
class PostprocessingConfig:

    # threshold for binary / multilabel
    threshold: float = 0.5

    # optional per-task thresholds
    task_thresholds: Optional[Dict[str, float]] = None

    # label mapping
    label_maps: Optional[Dict[str, Dict[int, str]]] = None

    # calibration
    apply_calibration: bool = False


# =========================================================
# CORE CLASS
# =========================================================

class Postprocessor:

    def __init__(self, config: Optional[PostprocessingConfig] = None):
        self.config = config or PostprocessingConfig()
        logger.info("Postprocessor initialized")

    # =====================================================
    # MAIN ENTRYPOINT
    # =====================================================

    def process(
        self,
        outputs: Dict[str, Any],
        *,
        task_types: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        """
        Process raw model outputs into final predictions.

        outputs:
        {
            task: {
                logits: np.ndarray
                probabilities: np.ndarray
            }
        }
        """

        results = {}

        for task, out in outputs.items():

            logits = out.get("logits")
            probs = out.get("probabilities")

            task_type = (
                task_types.get(task) if task_types else "multiclass"
            )

            if probs is None and logits is not None:
                probs = self._compute_probs(logits, task_type)

            preds = self._predict(probs, task, task_type)

            labels = self._map_labels(task, preds)

            confidence = self._confidence(probs, preds, task_type)

            results[task] = {
                "predictions": preds,
                "labels": labels,
                "confidence": confidence,
                "probabilities": probs,
                "logits": logits,
            }

        return results

    # =====================================================
    # PROBABILITIES
    # =====================================================

    def _compute_probs(self, logits, task_type):

        logits = np.asarray(logits)

        if task_type == "multiclass":
            e = np.exp(logits - np.max(logits, axis=1, keepdims=True))
            return e / (np.sum(e, axis=1, keepdims=True) + EPS)

        elif task_type in ("binary", "multilabel"):
            return 1 / (1 + np.exp(-logits))

        else:
            raise ValueError(f"Unknown task_type: {task_type}")

    # =====================================================
    # PREDICTIONS
    # =====================================================

    def _predict(self, probs, task, task_type):

        probs = np.asarray(probs)

        threshold = self._get_threshold(task)

        if task_type == "multiclass":
            return np.argmax(probs, axis=1)

        elif task_type == "binary":
            return (probs > threshold).astype(int)

        elif task_type == "multilabel":
            return (probs > threshold).astype(int)

        else:
            raise ValueError(f"Unknown task_type: {task_type}")

    # =====================================================
    # THRESHOLDS
    # =====================================================

    def _get_threshold(self, task):

        if self.config.task_thresholds and task in self.config.task_thresholds:
            return self.config.task_thresholds[task]

        return self.config.threshold

    # =====================================================
    # LABEL MAPPING
    # =====================================================

    def _map_labels(self, task, preds):

        if not self.config.label_maps:
            return preds

        mapping = self.config.label_maps.get(task)

        if not mapping:
            return preds

        return [mapping.get(int(p), str(p)) for p in preds]

    # =====================================================
    # CONFIDENCE
    # =====================================================

    def _confidence(self, probs, preds, task_type):

        probs = np.asarray(probs)

        if task_type == "multiclass":
            return np.max(probs, axis=1)

        elif task_type == "binary":
            return probs

        elif task_type == "multilabel":
            return np.max(probs, axis=1)

        return None

    # =====================================================
    # CALIBRATION HOOK (OPTIONAL)
    # =====================================================

    def apply_calibration(
        self,
        probs: np.ndarray,
        calibrator,
    ) -> np.ndarray:

        if not self.config.apply_calibration:
            return probs

        try:
            return calibrator.predict_proba(probs)
        except Exception as e:
            logger.warning("Calibration failed: %s", e)
            return probs