"""
File Name: prediction_output.py
Module: models.inference
Description:
    Defines structured prediction output objects used during inference in the
    TruthLens AI system. These dataclasses standardize the representation of
    model predictions, probabilities, logits, and optional metadata returned
    by the prediction pipeline.

    The module ensures that downstream components (evaluation pipelines,
    dashboards, APIs, logging systems, and explainability modules) receive
    consistent and strongly-typed prediction outputs.

    Supports both single-task and multi-task prediction scenarios.

Dependencies:
    dataclasses
    typing
    torch
Inputs:
    Model logits, probabilities, predictions, optional metadata
Outputs:
    Structured prediction objects
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional, Any

import torch


@dataclass
class TaskPrediction:
    """
    Structured prediction container for a single task.
    """

    logits: Optional[torch.Tensor] = None
    probabilities: Optional[torch.Tensor] = None
    predictions: Optional[torch.Tensor] = None
    confidence: Optional[torch.Tensor] = None
    metadata: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert task prediction to dictionary.
        """

        return {
            "logits": self.logits,
            "probabilities": self.probabilities,
            "predictions": self.predictions,
            "confidence": self.confidence,
            "metadata": self.metadata,
        }


@dataclass
class PredictionOutput:
    """
    Structured prediction output for the TruthLens inference pipeline.

    Attributes
    ----------
    tasks:
        Dictionary mapping task names to TaskPrediction objects.
    metadata:
        Optional global metadata such as model version, timestamp,
        or request identifiers.
    """

    tasks: Dict[str, TaskPrediction] = field(default_factory=dict)
    metadata: Optional[Dict[str, Any]] = None

    @staticmethod
    def _task_name_from_key(key: str) -> Optional[str]:
        suffixes = (
            "_logits",
            "_probabilities",
            "_calibrated_probabilities",
            "_predictions",
            "_confidence",
        )
        for suffix in suffixes:
            if key.endswith(suffix):
                return key[: -len(suffix)]
        return None

    @classmethod
    def from_raw_outputs(
        cls,
        raw_outputs: Dict[str, Any],
        *,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> "PredictionOutput":
        if not isinstance(raw_outputs, dict):
            raise TypeError("raw_outputs must be a dictionary")

        if "tasks" in raw_outputs and isinstance(raw_outputs["tasks"], dict):
            structured = cls(metadata=raw_outputs.get("metadata") or metadata)
            for task_name, task_values in raw_outputs["tasks"].items():
                if not isinstance(task_values, dict):
                    continue
                structured.tasks[task_name] = TaskPrediction(
                    logits=task_values.get("logits"),
                    probabilities=task_values.get("probabilities"),
                    predictions=task_values.get("predictions"),
                    confidence=task_values.get("confidence"),
                    metadata=task_values.get("metadata"),
                )
            return structured

        structured = cls(metadata=metadata)

        for key, value in raw_outputs.items():
            task_name = cls._task_name_from_key(key)
            if task_name is None:
                continue

            task = structured.tasks.get(task_name)
            if task is None:
                task = TaskPrediction()
                structured.tasks[task_name] = task

            if key.endswith("_logits"):
                task.logits = value if isinstance(value, torch.Tensor) else None
            elif key.endswith("_probabilities") and not key.endswith(
                "_calibrated_probabilities"
            ):
                task.probabilities = value if isinstance(value, torch.Tensor) else None
            elif key.endswith("_predictions"):
                task.predictions = value if isinstance(value, torch.Tensor) else None
            elif key.endswith("_confidence"):
                task.confidence = value if isinstance(value, torch.Tensor) else None

        return structured

    def add_task(
        self,
        task_name: str,
        logits: Optional[torch.Tensor] = None,
        probabilities: Optional[torch.Tensor] = None,
        predictions: Optional[torch.Tensor] = None,
        confidence: Optional[torch.Tensor] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Add prediction results for a specific task.
        """

        self.tasks[task_name] = TaskPrediction(
            logits=logits,
            probabilities=probabilities,
            predictions=predictions,
            confidence=confidence,
            metadata=metadata,
        )

    def get_task(self, task_name: str) -> TaskPrediction:
        """
        Retrieve prediction results for a specific task.
        """

        if task_name not in self.tasks:
            raise KeyError(f"Task '{task_name}' not found in prediction output")

        return self.tasks[task_name]

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert prediction output to dictionary representation.
        """

        result: Dict[str, Any] = {
            "tasks": {name: task.to_dict() for name, task in self.tasks.items()}
        }

        if self.metadata:
            result["metadata"] = self.metadata

        return result