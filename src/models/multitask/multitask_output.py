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

    logits: torch.Tensor
    probabilities: Optional[torch.Tensor] = None
    predictions: Optional[torch.Tensor] = None
    loss: Optional[torch.Tensor] = None

    def detach(self) -> "TaskOutput":
        """Detach tensors from computation graph for logging."""
        return TaskOutput(
            logits=self.logits.detach(),
            probabilities=self.probabilities.detach() if self.probabilities is not None else None,
            predictions=self.predictions.detach() if self.predictions is not None else None,
            loss=self.loss.detach() if self.loss is not None else None,
        )


@dataclass
class MultiTaskOutput:

    tasks: Dict[str, TaskOutput] = field(default_factory=dict)

    loss: Optional[torch.Tensor] = None
    task_losses: Optional[Dict[str, torch.Tensor]] = None
    metadata: Optional[Dict[str, Any]] = None

    @classmethod
    def from_model_outputs(cls, outputs: Dict[str, Any]) -> "MultiTaskOutput":

        if not isinstance(outputs, dict):
            raise TypeError("outputs must be a dictionary")

        if isinstance(outputs.get("multitask_output"), MultiTaskOutput):
            return outputs["multitask_output"]

        multitask = cls()

        for task_name, payload in outputs.items():

            if not isinstance(payload, dict):
                continue

            logits = payload.get("logits")

            if not isinstance(logits, torch.Tensor):
                continue

            multitask.add_task_output(
                task_name=task_name,
                logits=logits,
                probabilities=payload.get("probabilities"),
                predictions=payload.get("predictions"),
                loss=payload.get("loss"),
            )

        if isinstance(outputs.get("loss"), torch.Tensor):
            multitask.loss = outputs["loss"]

        task_losses = outputs.get("task_losses") or outputs.get("loss_breakdown")

        if isinstance(task_losses, dict):
            multitask.task_losses = task_losses

        return multitask

    def add_task_output(
        self,
        task_name: str,
        logits: torch.Tensor,
        probabilities: Optional[torch.Tensor] = None,
        predictions: Optional[torch.Tensor] = None,
        loss: Optional[torch.Tensor] = None,
    ) -> None:

        if not isinstance(logits, torch.Tensor):
            raise TypeError("logits must be torch.Tensor")

        self.tasks[task_name] = TaskOutput(
            logits=logits,
            probabilities=probabilities,
            predictions=predictions,
            loss=loss,
        )

    def get_logits(self, task_name: str) -> torch.Tensor:
        return self.tasks[task_name].logits

    def get_predictions(self, task_name: str) -> Optional[torch.Tensor]:
        return self.tasks[task_name].predictions

    def get_probabilities(self, task_name: str) -> Optional[torch.Tensor]:
        return self.tasks[task_name].probabilities

    def get_task_loss(self, task_name: str) -> Optional[torch.Tensor]:
        return self.tasks[task_name].loss

    def to_dict(self, detach: bool = True) -> Dict[str, Any]:
        """Convert outputs to serializable dictionary."""

        result: Dict[str, Any] = {}

        for task_name, task_output in self.tasks.items():

            if detach:
                task_output = task_output.detach()

            result[task_name] = {
                "logits": task_output.logits,
                "probabilities": task_output.probabilities,
                "predictions": task_output.predictions,
                "loss": task_output.loss,
            }

        result["loss"] = self.loss.detach() if detach and self.loss is not None else self.loss

        if self.task_losses is not None:
            if detach:
                result["task_losses"] = {
                    key: value.detach() if isinstance(value, torch.Tensor) else value
                    for key, value in self.task_losses.items()
                }
            else:
                result["task_losses"] = self.task_losses

        if self.metadata:
            result["metadata"] = self.metadata

        return result

    def to_flat_prediction_dict(self) -> Dict[str, Any]:

        flat: Dict[str, Any] = {}

        for task_name, task_output in self.tasks.items():

            flat[f"{task_name}_logits"] = task_output.logits

            if task_output.probabilities is not None:
                flat[f"{task_name}_probabilities"] = task_output.probabilities

            if task_output.predictions is not None:
                flat[f"{task_name}_predictions"] = task_output.predictions

        if self.loss is not None:
            flat["loss"] = self.loss

        if self.task_losses is not None:
            flat["task_losses"] = self.task_losses

        if self.metadata:
            flat["metadata"] = self.metadata

        return flat