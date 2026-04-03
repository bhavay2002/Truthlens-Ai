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