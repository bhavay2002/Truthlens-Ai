"""
File Name: multitask_base_model.py
Module: models.base
Description:
    Defines the abstract base class for multi-task learning models in the
    TruthLens ML framework. This class extends the BaseModel abstraction and
    provides standardized interfaces for handling multiple prediction tasks,
    task-specific heads, loss computation, and prediction outputs. It is designed
    to support transformer-based encoders and flexible task heads for research
    experimentation and production deployment.

Dependencies:
    torch
    torch.nn
    torch.nn.functional
    typing
    logging
    models.base.base_model
Inputs:
    Encoded representations and optional task-specific labels.
Outputs:
    Dictionary containing logits, probabilities, predictions, and losses for
    each task.
"""

from __future__ import annotations

import logging
from abc import abstractmethod
from typing import Any, Dict, Optional

import torch
import torch.nn.functional as F
from torch import nn

from .base_model import BaseModel

logger = logging.getLogger(__name__)


class MultiTaskBaseModel(BaseModel):
    """
    Base class for multi-task models.

    This abstraction supports models that produce predictions for multiple tasks
    simultaneously (e.g., bias detection, emotion classification, propaganda
    detection). Each task is associated with its own prediction head and loss
    function.
    """

    def __init__(self, task_configs: Dict[str, Dict[str, Any]]) -> None:
        """
        Initializes the multi-task model.

        Args:
            task_configs:
                Dictionary defining task configurations.

                Example:
                {
                    "bias": {"num_classes": 3, "type": "classification"},
                    "emotion": {"num_classes": 28, "type": "multilabel"},
                    "propaganda": {"num_classes": 1, "type": "binary"}
                }
        """
        super().__init__()

        if not isinstance(task_configs, dict) or not task_configs:
            raise ValueError("task_configs must be a non-empty dictionary")

        self.task_configs = task_configs
        self.task_heads: nn.ModuleDict = nn.ModuleDict()
        self.loss_functions: Dict[str, nn.Module] = {}

    @abstractmethod
    def encode(self, *inputs: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        """
        Encodes raw inputs into shared feature representations.

        Must be implemented by subclasses.

        Returns:
            Tensor of shape (batch_size, hidden_dim)
        """
        raise NotImplementedError("Subclasses must implement encode().")

    def register_task_head(
        self,
        task_name: str,
        head: nn.Module,
        loss_fn: nn.Module,
    ) -> None:
        """
        Registers a task-specific prediction head.

        Args:
            task_name:
                Name of the task.
            head:
                Neural network module used for prediction.
            loss_fn:
                Loss function associated with the task.
        """
        if task_name in self.task_heads:
            raise ValueError(f"Task '{task_name}' already registered")

        self.task_heads[task_name] = head
        self.loss_functions[task_name] = loss_fn

        logger.info("Registered task head: %s", task_name)

    def forward(
        self,
        *inputs: torch.Tensor,
        labels: Optional[Dict[str, torch.Tensor]] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """
        Executes forward pass for all registered tasks.

        Args:
            *inputs:
                Input tensors.
            labels:
                Optional dictionary mapping task names to ground truth labels.

        Returns:
            Dictionary containing task outputs and optional losses.
        """
        shared_features = self.encode(*inputs, **kwargs)

        if shared_features.dim() != 2:
            raise ValueError(
                f"Encoded features must be 2D (batch_size, hidden_dim), "
                f"got shape {shared_features.shape}"
            )

        outputs: Dict[str, Any] = {"tasks": {}}
        total_loss: Optional[torch.Tensor] = None

        for task_name, head in self.task_heads.items():
            logits = head(shared_features)

            task_config = self.task_configs.get(task_name, {})
            task_type = task_config.get("type", "classification")

            if task_type == "multilabel":
                probs = torch.sigmoid(logits)
                preds = (probs > 0.5).long()
            else:
                probs = F.softmax(logits, dim=-1)
                preds = torch.argmax(probs, dim=-1)

            task_output: Dict[str, torch.Tensor] = {
                "logits": logits,
                "probabilities": probs,
                "predictions": preds,
            }

            if labels and task_name in labels:
                loss_fn = self.loss_functions.get(task_name)
                if loss_fn is None:
                    raise RuntimeError(f"No loss function registered for {task_name}")

                task_labels = labels[task_name]

                if task_type == "multilabel":
                    task_labels = task_labels.float()

                task_loss = loss_fn(logits, task_labels)

                task_output["loss"] = task_loss

                if total_loss is None:
                    total_loss = task_loss
                else:
                    total_loss = total_loss + task_loss

            outputs["tasks"][task_name] = task_output

        if total_loss is not None:
            outputs["loss"] = total_loss

        return outputs

    @torch.no_grad()
    def predict(
        self,
        *inputs: torch.Tensor,
        **kwargs: Any,
    ) -> Dict[str, Dict[str, torch.Tensor]]:
        """
        Performs inference for all tasks.

        Args:
            *inputs:
                Input tensors.

        Returns:
            Dictionary containing predictions and probabilities per task.
        """
        self.eval()

        outputs = self.forward(*inputs, **kwargs)

        predictions: Dict[str, Dict[str, torch.Tensor]] = {}

        for task_name, task_output in outputs["tasks"].items():
            predictions[task_name] = {
                "predictions": task_output["predictions"],
                "probabilities": task_output["probabilities"],
            }

        return predictions

    def get_tasks(self) -> Dict[str, Dict[str, Any]]:
        """
        Returns task configuration.

        Returns:
            Task configuration dictionary.
        """
        return self.task_configs