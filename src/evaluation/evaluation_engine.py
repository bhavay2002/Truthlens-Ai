# src/evaluation/evaluation_engine.py

from __future__ import annotations

import logging
from typing import Dict, Any, Optional

from torch.utils.data import DataLoader

from src.evaluation.metrics_engine import MetricsEngine
from src.evaluation.prediction_collector import collect_all_tasks

logger = logging.getLogger(__name__)


# =========================================================
# ENGINE
# =========================================================

class EvaluationEngine:
    """
    Lightweight evaluation orchestrator.

    Responsibilities:
    - collect predictions
    - compute metrics
    - return standardized output

    DOES NOT:
    - run heavy analysis (handled by evaluation_pipeline)
    """

    def __init__(
        self,
        metrics_engine: Optional[MetricsEngine] = None,
        task_types: Optional[Dict[str, str]] = None,
    ):
        self.metrics_engine = metrics_engine or MetricsEngine()
        self.task_types = task_types or {}

        logger.info("EvaluationEngine initialized")

    # =====================================================
    # MAIN ENTRY (USED BY TRAINER)
    # =====================================================

    def evaluate(
        self,
        model,
        dataloader: DataLoader,
        *,
        device=None,
    ) -> Dict[str, Any]:

        logger.info("Running evaluation...")

        # -------------------------
        # STEP 1: COLLECT PREDICTIONS
        # -------------------------
        predictions = self._collect_predictions(
            model=model,
            dataloader=dataloader,
            device=device,
        )

        # -------------------------
        # STEP 2: COMPUTE METRICS
        # -------------------------
        metrics = self.metrics_engine.compute_multitask(
            predictions=predictions,
            task_types=self.task_types,
        )

        # -------------------------
        # OUTPUT FORMAT (STANDARD)
        # -------------------------
        return {
            "metrics": metrics,
            "val_loss": self._extract_val_loss(metrics),
        }

    # =====================================================
    # INTERNAL
    # =====================================================

    def _collect_predictions(
        self,
        model,
        dataloader: DataLoader,
        device=None,
    ) -> Dict[str, Dict[str, Any]]:

        return collect_all_tasks(
            model=model,
            dataloader=dataloader,
            device=device,
        )

    # =====================================================
    # HELPER
    # =====================================================

    def _extract_val_loss(self, metrics: Dict[str, Any]) -> float:
        """
        Extract a scalar for Trainer early stopping.
        """

        agg = metrics.get("__aggregate__")

        if agg and "log_loss" in agg:
            return float(agg["log_loss"])

        # fallback
        for task, m in metrics.items():
            if isinstance(m, dict) and "log_loss" in m:
                return float(m["log_loss"])

        return 0.0