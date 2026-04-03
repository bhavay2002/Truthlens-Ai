"""
File Name: multitask_head.py
Module: models.heads
Description:
    Implements a flexible multi-task prediction head used in the TruthLens AI
    system. This module manages multiple task-specific heads (classification,
    regression, multilabel, etc.) and orchestrates forward execution, prediction
    aggregation, and optional loss computation.

    The multitask head allows a shared encoder representation to feed multiple
    prediction objectives simultaneously, which is essential for architectures
    like the TruthLens multi-task model predicting bias, ideology, propaganda,
    narrative structure, and emotion.

    Each task head is registered with its own configuration, output dimension,
    and optional loss function.

Dependencies:
    logging
    typing
    torch
    torch.nn

Inputs:
    Shared encoder embeddings (batch_size, hidden_dim)
    Optional task label dictionary

Outputs:
    Dictionary containing:
        - task logits
        - probabilities
        - predictions
        - optional task losses
        - total_loss
"""

from __future__ import annotations

import logging
from typing import Dict, Optional, Any

import torch
import torch.nn as nn


logger = logging.getLogger(__name__)


class MultiTaskHead(nn.Module):
    """
    Multi-task prediction head.

    This module routes shared encoder representations into multiple
    task-specific heads and aggregates outputs.
    """

    def __init__(self) -> None:
        super().__init__()

        self.task_heads: nn.ModuleDict = nn.ModuleDict()
        self.loss_fns: Dict[str, nn.Module] = {}

    def register_task(
        self,
        task_name: str,
        head: nn.Module,
        loss_fn: Optional[nn.Module] = None,
    ) -> None:
        """
        Register a task head.

        Args:
            task_name:
                Unique name of the task.
            head:
                Neural module implementing prediction head.
            loss_fn:
                Optional loss function for the task.
        """

        if not isinstance(task_name, str) or not task_name.strip():
            raise ValueError("task_name must be a valid string")

        if task_name in self.task_heads:
            raise ValueError(f"Task '{task_name}' already registered")

        if not isinstance(head, nn.Module):
            raise TypeError("head must be a torch.nn.Module")

        self.task_heads[task_name] = head

        if loss_fn is not None:
            self.loss_fns[task_name] = loss_fn

        logger.info("Registered multitask head for task: %s", task_name)

    def forward(
        self,
        features: torch.Tensor,
        labels: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Dict[str, Any]:
        """
        Forward pass through all task heads.

        Args:
            features:
                Shared encoder embeddings (batch_size, hidden_dim)
            labels:
                Optional dictionary mapping task names to label tensors.

        Returns:
            Aggregated task outputs.
        """

        if features is None:
            raise ValueError("features cannot be None")

        if features.dim() != 2:
            raise ValueError(
                f"Expected features shape (batch_size, hidden_dim), got {features.shape}"
            )

        outputs: Dict[str, Any] = {"tasks": {}}
        total_loss: Optional[torch.Tensor] = None

        for task_name, head in self.task_heads.items():

            head_output = head(features)

            if not isinstance(head_output, dict):
                # assume logits-only output
                logits = head_output
                task_output = {"logits": logits}
            else:
                task_output = head_output
                logits = task_output.get("logits")

            if logits is None:
                raise RuntimeError(
                    f"Task head '{task_name}' must produce logits or logits field"
                )

            outputs["tasks"][task_name] = task_output

            if labels and task_name in labels:

                if task_name not in self.loss_fns:
                    raise RuntimeError(
                        f"No loss function registered for task '{task_name}'"
                    )

                loss_fn = self.loss_fns[task_name]
                task_labels = labels[task_name]

                loss = loss_fn(logits, task_labels)

                outputs["tasks"][task_name]["loss"] = loss

                if total_loss is None:
                    total_loss = loss
                else:
                    total_loss = total_loss + loss

        if total_loss is not None:
            outputs["total_loss"] = total_loss

        return outputs

    @torch.no_grad()
    def predict(self, features: torch.Tensor) -> Dict[str, Any]:
        """
        Run inference for all tasks.

        Args:
            features:
                Shared encoder embeddings.

        Returns:
            Task prediction dictionary.
        """

        self.eval()

        outputs = self.forward(features)

        predictions: Dict[str, Any] = {}

        for task_name, task_output in outputs["tasks"].items():

            preds = task_output.get("predictions")
            probs = task_output.get("probabilities")

            predictions[task_name] = {
                "predictions": preds,
                "probabilities": probs,
            }

        return predictions

    def get_tasks(self) -> Dict[str, nn.Module]:
        """
        Return registered task heads.
        """

        return dict(self.task_heads)