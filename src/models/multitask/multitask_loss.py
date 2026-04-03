"""
File Name: multitask_loss.py
Module: models.multitask
Description:
    Implements loss computation utilities for the TruthLens multi-task model.
    The module provides a flexible multi-task loss manager capable of handling
    heterogeneous objectives including:

        • binary classification
        • multi-class classification
        • multi-label classification
        • regression (optional future tasks)

    The implementation supports task-specific loss functions, configurable
    task weighting, automatic validation of label shapes, and aggregation
    of task losses into a unified training objective.

    This module is designed for production ML pipelines and research
    experimentation where different tasks may require different loss
    formulations.

Dependencies:
    logging
    typing
    dataclasses
    torch
    torch.nn
Inputs:
    logits dictionary per task
    label dictionary per task
Outputs:
    total loss and per-task loss dictionary
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


@dataclass
class TaskLossConfig:
    """
    Configuration describing a single task loss.
    """

    task_type: str
    weight: float = 1.0


class MultiTaskLoss(nn.Module):
    """
    Multi-task loss manager.

    Supports multiple task types simultaneously and aggregates
    losses into a single scalar objective.
    """

    VALID_TASK_TYPES = {
        "binary",
        "multi_class",
        "multi_label",
        "regression",
    }

    def __init__(
        self,
        task_configs: Dict[str, TaskLossConfig],
    ) -> None:
        """
        Initialize multi-task loss manager.

        Args:
            task_configs:
                Mapping of task_name -> TaskLossConfig
        """

        super().__init__()

        if not isinstance(task_configs, dict) or not task_configs:
            raise ValueError("task_configs must be a non-empty dictionary")

        self.task_configs = task_configs

        self.loss_functions: Dict[str, nn.Module] = {}

        for task_name, config in task_configs.items():

            if config.task_type not in self.VALID_TASK_TYPES:
                raise ValueError(
                    f"Invalid task_type '{config.task_type}' for task '{task_name}'"
                )

            if config.task_type == "multi_class":
                self.loss_functions[task_name] = nn.CrossEntropyLoss()

            elif config.task_type == "binary":
                self.loss_functions[task_name] = nn.BCEWithLogitsLoss()

            elif config.task_type == "multi_label":
                self.loss_functions[task_name] = nn.BCEWithLogitsLoss()

            elif config.task_type == "regression":
                self.loss_functions[task_name] = nn.MSELoss()

        logger.info("MultiTaskLoss initialized with tasks: %s", list(task_configs.keys()))

    def forward(
        self,
        logits: Dict[str, torch.Tensor],
        labels: Dict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Compute total multi-task loss.

        Args:
            logits:
                Dictionary of model logits per task.
            labels:
                Dictionary of ground-truth labels per task.

        Returns:
            total_loss, task_loss_dict
        """

        if not isinstance(logits, dict):
            raise TypeError("logits must be a dictionary")

        if not isinstance(labels, dict):
            raise TypeError("labels must be a dictionary")

        task_losses: Dict[str, torch.Tensor] = {}

        total_loss: Optional[torch.Tensor] = None

        for task_name, task_config in self.task_configs.items():

            if task_name not in logits:
                continue

            if task_name not in labels:
                continue

            task_logits = logits[task_name]
            task_labels = labels[task_name]

            loss_fn = self.loss_functions[task_name]

            task_type = task_config.task_type

            if task_type == "multi_class":

                if task_labels.dim() == 2:
                    task_labels = task_labels.argmax(dim=1)

                loss = loss_fn(task_logits, task_labels.long())

            elif task_type == "binary":

                if task_labels.dim() == 1:
                    task_labels = task_labels.unsqueeze(1)

                loss = loss_fn(task_logits, task_labels.float())

            elif task_type == "multi_label":

                loss = loss_fn(task_logits, task_labels.float())

            elif task_type == "regression":

                loss = loss_fn(task_logits, task_labels.float())

            else:
                raise RuntimeError(f"Unsupported task type: {task_type}")

            weighted_loss = loss * task_config.weight

            task_losses[task_name] = weighted_loss

            if total_loss is None:
                total_loss = weighted_loss
            else:
                total_loss = total_loss + weighted_loss

        if total_loss is None:
            raise RuntimeError("No task losses were computed")

        return total_loss, task_losses

    def get_task_names(self) -> Dict[str, TaskLossConfig]:
        """
        Return task configuration dictionary.
        """

        return self.task_configs