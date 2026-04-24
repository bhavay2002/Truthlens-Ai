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

_MULTICLASS_TASKS: frozenset = frozenset({"bias", "ideology", "propaganda"})
_MULTILABEL_TASKS: frozenset = frozenset({"narrative", "narrative_frame", "emotion"})
_VALID_TASKS: frozenset = _MULTICLASS_TASKS | _MULTILABEL_TASKS


def _validate_task(task: str) -> None:
    if task not in _VALID_TASKS:
        raise ValueError(
            f"Unknown task {task!r}. Valid tasks: {sorted(_VALID_TASKS)}"
        )


def _compute_confidence(
    task: str,
    probabilities: Optional[torch.Tensor],
) -> Optional[torch.Tensor]:
    if probabilities is None:
        return None
    if task in _MULTICLASS_TASKS:
        return probabilities.max(dim=-1).values
    if task in _MULTILABEL_TASKS:
        return probabilities.mean(dim=-1)
    return None


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

    @classmethod
    def from_raw_outputs(
        cls,
        raw_outputs: Dict[str, Any],
        *,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> "PredictionOutput":
        structured = cls(metadata=metadata)
        tasks = structured.tasks
        get_task = tasks.get

        field_map = {
            "logits": "logits",
            "probabilities": "probabilities",
            "predictions": "predictions",
            "confidence": "confidence",
        }

        for key, value in raw_outputs.items():
            split = key.rsplit("_", 1)
            if len(split) != 2:
                continue

            task_name, field = split
            if not task_name:
                continue

            task = get_task(task_name)
            if task is None:
                task = TaskPrediction()
                tasks[task_name] = task

            attr = field_map.get(field)
            if attr is not None:
                setattr(task, attr, value)

        return structured

    @classmethod
    def from_flat(
        cls,
        flat_outputs: Dict[str, Any],
        *,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> "PredictionOutput":
        if not isinstance(flat_outputs, dict):
            raise TypeError("flat_outputs must be a dictionary")

        tasks_payload = flat_outputs.get("tasks")
        if isinstance(tasks_payload, dict):
            structured = cls(metadata=metadata)
            for task_name, values in tasks_payload.items():
                if not isinstance(values, dict):
                    continue
                structured.add_task(
                    task_name=task_name,
                    logits=values.get("logits"),
                    probabilities=values.get("probabilities"),
                    predictions=values.get("predictions"),
                    confidence=values.get("confidence"),
                    metadata=values.get("metadata"),
                )
            return structured

        return cls.from_raw_outputs(flat_outputs, metadata=metadata)

    @classmethod
    def fast_from_raw(cls, raw_outputs: Dict[str, Any]) -> Dict[str, Any]:
        return raw_outputs

    @classmethod
    def from_single_task(
        cls,
        task: str,
        outputs: Dict[str, Any],
        *,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> "PredictionOutput":
        """Build a ``PredictionOutput`` from a flat single-task output dict.

        Avoids the string-splitting logic of ``from_raw_outputs`` and
        auto-computes confidence from the probability tensor.
        """
        _validate_task(task)
        structured = cls(metadata=metadata)
        probs = outputs.get("probabilities")
        structured.add_task(
            task_name=task,
            logits=outputs.get("logits"),
            probabilities=probs,
            predictions=outputs.get("predictions"),
            confidence=outputs.get("confidence") or _compute_confidence(task, probs),
        )
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
        Auto-computes confidence from probabilities when not supplied.
        """
        if confidence is None and probabilities is not None:
            try:
                confidence = _compute_confidence(task_name, probabilities)
            except ValueError:
                pass

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

    def to_lightweight(self) -> Dict[str, Any]:
        """Return a minimal dict: task name → predictions tensor only."""
        return {name: task.predictions for name, task in self.tasks.items()}