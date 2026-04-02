"""
File Name: evaluator.py
Module: TruthLens AI - Evaluation Core
Description:
    Centralized evaluation engine used across the TruthLens AI system.
    Supports binary, multi-class, multi-label, and multi-task evaluation.
    Provides consistent metric computation, structured logging, validation,
    and aggregation utilities for research and production pipelines.
Dependencies:
    logging
    typing
    numpy
    src.evaluation.metrics
Inputs:
    y_true: Ground truth labels
    y_pred: Predicted labels
    y_proba: Optional prediction probabilities
    results: Dictionary containing task-level metric outputs
Outputs:
    Dictionary containing evaluation metrics and aggregated summaries
"""

from __future__ import annotations

import logging
from typing import Dict, Any, Iterable, Optional

import numpy as np

from .metrics import (
    compute_classification_metrics,
    compute_multilabel_metrics,
)

logger = logging.getLogger(__name__)


class Evaluator:
    """
    Central evaluation engine responsible for computing metrics
    across different task types in the TruthLens system.
    """

    @staticmethod
    def _validate_inputs(
        y_true: Iterable,
        y_pred: Iterable,
        y_proba: Optional[Iterable] = None
    ) -> None:
        """
        Validate evaluation inputs.
        """

        if y_true is None or y_pred is None:
            raise ValueError("y_true and y_pred must not be None.")

        y_true_arr = np.asarray(y_true)
        y_pred_arr = np.asarray(y_pred)

        if y_true_arr.shape[0] == 0:
            raise ValueError("y_true is empty.")

        if y_true_arr.shape != y_pred_arr.shape:
            raise ValueError(
                f"Shape mismatch between y_true {y_true_arr.shape} "
                f"and y_pred {y_pred_arr.shape}."
            )

        if y_proba is not None:
            y_proba_arr = np.asarray(y_proba)
            if y_proba_arr.shape[0] != y_true_arr.shape[0]:
                raise ValueError(
                    "y_proba must have the same number of samples as y_true."
                )

    @staticmethod
    def classification(
        y_true: Iterable,
        y_pred: Iterable,
        y_proba: Optional[Iterable] = None
    ) -> Dict[str, Any]:
        """
        Evaluate binary or multi-class classification models.
        """

        logger.info("Running classification evaluation")

        Evaluator._validate_inputs(y_true, y_pred, y_proba)

        try:
            metrics = compute_classification_metrics(
                y_true=y_true,
                y_pred=y_pred,
                y_proba=y_proba
            )

            logger.info("Classification evaluation completed")
            return metrics

        except Exception as exc:
            logger.exception("Classification evaluation failed")
            raise RuntimeError("Classification evaluation failed") from exc

    @staticmethod
    def multilabel(
        y_true: Iterable,
        y_pred: Iterable
    ) -> Dict[str, Any]:
        """
        Evaluate multi-label classification models.
        """

        logger.info("Running multilabel evaluation")

        Evaluator._validate_inputs(y_true, y_pred)

        try:
            metrics = compute_multilabel_metrics(
                y_true=y_true,
                y_pred=y_pred
            )

            logger.info("Multilabel evaluation completed")
            return metrics

        except Exception as exc:
            logger.exception("Multilabel evaluation failed")
            raise RuntimeError("Multilabel evaluation failed") from exc

    @staticmethod
    def multitask(results: Dict[str, Dict[str, Any]]) -> Dict[str, float]:
        """
        Aggregate metrics across multiple tasks.

        Example input:
        {
            "bias": {"accuracy": 0.82, "f1": 0.80},
            "emotion": {"accuracy": 0.78, "f1": 0.76}
        }
        """

        if not isinstance(results, dict):
            raise TypeError("results must be a dictionary.")

        logger.info("Running multitask evaluation aggregation")

        metrics = ["accuracy", "f1", "precision", "recall"]

        summary: Dict[str, float] = {}

        for metric in metrics:

            values = []

            for task_name, task_res in results.items():

                if not isinstance(task_res, dict):
                    logger.warning(
                        "Invalid metrics format for task %s. Skipping.",
                        task_name
                    )
                    continue

                val = task_res.get(metric)

                if val is None:
                    continue

                try:
                    values.append(float(val))
                except (TypeError, ValueError):
                    logger.warning(
                        "Metric %s for task %s is not numeric.",
                        metric,
                        task_name
                    )

            if values:
                summary[f"avg_{metric}"] = float(np.mean(values))
                summary[f"std_{metric}"] = float(np.std(values))

        logger.info("Multitask aggregation completed")

        return summary