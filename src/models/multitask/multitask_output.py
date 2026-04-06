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

    @classmethod
    def from_model_outputs(
        cls,
        outputs: Dict[str, Any],
    ) -> "MultiTaskOutput":
        if not isinstance(outputs, dict):
            raise TypeError("outputs must be a dictionary")

        if isinstance(outputs.get("multitask_output"), MultiTaskOutput):
            return outputs["multitask_output"]

        multitask = cls()

        for task_name, task_payload in outputs.items():
            if not isinstance(task_payload, dict):
                continue
            logits = task_payload.get("logits")
            if not isinstance(logits, torch.Tensor):
                continue

            multitask.add_task_output(
                task_name=task_name,
                logits=logits,
                probabilities=(
                    task_payload.get("probabilities")
                    if isinstance(task_payload.get("probabilities"), torch.Tensor)
                    else None
                ),
                predictions=(
                    task_payload.get("predictions")
                    if isinstance(task_payload.get("predictions"), torch.Tensor)
                    else None
                ),
                loss=(
                    task_payload.get("loss")
                    if isinstance(task_payload.get("loss"), torch.Tensor)
                    else None
                ),
            )

        if isinstance(outputs.get("loss"), torch.Tensor):
            multitask.loss = outputs.get("loss")

        task_losses = outputs.get("task_losses") or outputs.get("loss_breakdown")
        if isinstance(task_losses, dict):
            multitask.task_losses = {
                key: value for key, value in task_losses.items()
                if isinstance(value, torch.Tensor)
            }

        return multitask

    def to_flat_prediction_dict(self) -> Dict[str, Any]:
        flattened: Dict[str, Any] = {}

        for task_name, task_output in self.tasks.items():
            flattened[f"{task_name}_logits"] = task_output.logits
            if task_output.probabilities is not None:
                flattened[f"{task_name}_probabilities"] = task_output.probabilities
            if task_output.predictions is not None:
                flattened[f"{task_name}_predictions"] = task_output.predictions

        if self.loss is not None:
            flattened["loss"] = self.loss
        if self.task_losses is not None:
            flattened["task_losses"] = self.task_losses

        if self.metadata:
            flattened["metadata"] = self.metadata

        return flattened

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