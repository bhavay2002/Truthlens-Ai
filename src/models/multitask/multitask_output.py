"""
File Name: multitask_output.py
Module: models.multitask
Description:
    Defines structured output objects for the TruthLens multi-task model.
    The module standardizes how predictions, logits, probabilities, and
    task-specific losses are returned from the multi-task architecture.

    Structured outputs simplify:
        • training loop integration
        • evaluation pipelines
        • experiment logging
        • API responses
        • explainability modules

    The design follows patterns used in modern ML frameworks such as
    HuggingFace Transformers where model outputs are returned as structured
    dataclasses rather than loosely defined dictionaries.

Dependencies:
    dataclasses
    typing
    torch
Inputs:
    Model logits and probabilities produced by the multi-task model
Outputs:
    Structured multi-task model outputs
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional, Any

import torch


@dataclass
class TaskOutput:
    """
    Output container for a single task.
    """

    logits: torch.Tensor
    probabilities: Optional[torch.Tensor] = None
    predictions: Optional[torch.Tensor] = None
    loss: Optional[torch.Tensor] = None


@dataclass
class MultiTaskOutput:
    """
    Structured output returned by the TruthLens multi-task model.

    Attributes
    ----------
    tasks:
        Dictionary mapping task names to TaskOutput objects.
    loss:
        Total aggregated loss across tasks.
    task_losses:
        Dictionary containing per-task loss values.
    metadata:
        Optional metadata useful for logging or debugging.
    """

    tasks: Dict[str, TaskOutput] = field(default_factory=dict)

    loss: Optional[torch.Tensor] = None

    task_losses: Optional[Dict[str, torch.Tensor]] = None

    metadata: Optional[Dict[str, Any]] = None

    def add_task_output(
        self,
        task_name: str,
        logits: torch.Tensor,
        probabilities: Optional[torch.Tensor] = None,
        predictions: Optional[torch.Tensor] = None,
        loss: Optional[torch.Tensor] = None,
    ) -> None:
        """
        Add output for a specific task.

        Parameters
        ----------
        task_name:
            Name of the task.
        logits:
            Raw model logits.
        probabilities:
            Probabilities derived from logits.
        predictions:
            Final predicted labels.
        loss:
            Optional task loss.
        """

        self.tasks[task_name] = TaskOutput(
            logits=logits,
            probabilities=probabilities,
            predictions=predictions,
            loss=loss,
        )

    def get_logits(self, task_name: str) -> torch.Tensor:
        """
        Retrieve logits for a specific task.
        """

        return self.tasks[task_name].logits

    def get_probabilities(self, task_name: str) -> Optional[torch.Tensor]:
        """
        Retrieve probabilities for a specific task.
        """

        return self.tasks[task_name].probabilities

    def get_predictions(self, task_name: str) -> Optional[torch.Tensor]:
        """
        Retrieve predictions for a specific task.
        """

        return self.tasks[task_name].predictions

    def get_task_loss(self, task_name: str) -> Optional[torch.Tensor]:
        """
        Retrieve loss for a specific task.
        """

        return self.tasks[task_name].loss

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert output structure to a serializable dictionary.
        Useful for logging, monitoring, or API responses.
        """

        result: Dict[str, Any] = {}

        for task_name, task_output in self.tasks.items():
            result[task_name] = {
                "logits": task_output.logits,
                "probabilities": task_output.probabilities,
                "predictions": task_output.predictions,
                "loss": task_output.loss,
            }

        result["loss"] = self.loss
        result["task_losses"] = self.task_losses

        if self.metadata:
            result["metadata"] = self.metadata

        return result