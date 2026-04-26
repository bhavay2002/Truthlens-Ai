from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, Any, Optional

import numpy as np

from src.evaluation.metrics_engine import (
    compute_classification_metrics,
    compute_multilabel_metrics,
)

logger = logging.getLogger(__name__)


# =========================================================
# CONFIG
# =========================================================

@dataclass
class MetricsEngineConfig:
    """
    Controls evaluation behavior.
    """

    default_threshold: float = 0.5
    enable_confidence_weighting: bool = False
    return_per_task: bool = True
    aggregate: bool = True


# =========================================================
# ENGINE
# =========================================================

class MetricsEngine:
    """
    Unified metrics computation engine.

    Handles:
    - classification
    - multilabel
    - multi-task
    - aggregation
    """

    def __init__(self, config: Optional[MetricsEngineConfig] = None):
        self.config = config or MetricsEngineConfig()

        logger.info("MetricsEngine initialized")

    # =====================================================
    # SINGLE TASK
    # =====================================================

    def compute_task(
        self,
        *,
        y_true,
        y_pred,
        y_proba=None,
        task_type: str,
        threshold: Optional[float] = None,
        confidence=None,
    ) -> Dict[str, Any]:

        threshold = threshold or self.config.default_threshold

        if task_type == "classification":
            return compute_classification_metrics(
                y_true=y_true,
                y_pred=y_pred,
                y_proba=y_proba,
                threshold=threshold,
                confidence=confidence if self.config.enable_confidence_weighting else None,
            )

        elif task_type == "multilabel":
            return compute_multilabel_metrics(
                y_true=y_true,
                y_pred=y_pred,
                y_proba=y_proba,
                threshold=threshold,
            )

        else:
            raise ValueError(f"Unknown task_type: {task_type}")

    # =====================================================
    # MULTI-TASK (CORE)
    # =====================================================

    def compute_multitask(
        self,
        *,
        predictions: Dict[str, Dict[str, Any]],
        task_types: Dict[str, str],
        thresholds: Optional[Dict[str, float]] = None,
    ) -> Dict[str, Any]:

        results: Dict[str, Any] = {}
        aggregated: Dict[str, float] = {}

        for task, data in predictions.items():

            if task not in task_types:
                logger.warning(f"Missing task type for {task}")
                continue

            task_type = task_types[task]
            threshold = None

            if thresholds and task in thresholds:
                threshold = thresholds[task]

            metrics = self.compute_task(
                y_true=data["y_true"],
                y_pred=data["y_pred"],
                y_proba=data.get("y_proba"),
                task_type=task_type,
                threshold=threshold,
                confidence=data.get("confidence"),
            )

            results[task] = metrics

            # -------------------------
            # AGGREGATION
            # -------------------------
            if self.config.aggregate:
                self._accumulate_metrics(aggregated, metrics)

        if self.config.aggregate:
            aggregated = self._finalize_aggregation(aggregated, len(results))
            results["__aggregate__"] = aggregated

        return results

    # =====================================================
    # AGGREGATION
    # =====================================================

    def _accumulate_metrics(self, agg: Dict[str, float], metrics: Dict[str, Any]):

        for k, v in metrics.items():

            if not isinstance(v, (int, float)):
                continue

            agg[k] = agg.get(k, 0.0) + float(v)

    def _finalize_aggregation(self, agg: Dict[str, float], n: int):

        if n == 0:
            return agg

        return {k: v / n for k, v in agg.items()}