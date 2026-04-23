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
    ignore_index: int = -100  # used for multi_class tasks


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
        *,
        strict: bool = True,
    ) -> None:
        super().__init__()

        if not isinstance(task_configs, dict) or not task_configs:
            raise ValueError("task_configs must be a non-empty dictionary")

        self.task_configs = task_configs
        self.strict = strict

        #  M-LOSS-1: Register modules correctly
        self.loss_functions = nn.ModuleDict()

        for task_name, config in task_configs.items():

            if not isinstance(config, TaskLossConfig):
                raise ValueError(f"{task_name}: invalid TaskLossConfig")

            if config.task_type not in self.VALID_TASK_TYPES:
                raise ValueError(
                    f"Invalid task_type '{config.task_type}' for task '{task_name}'"
                )

            if config.task_type == "multi_class":
                self.loss_functions[task_name] = nn.CrossEntropyLoss(
                    ignore_index=config.ignore_index
                )

            elif config.task_type in {"binary", "multi_label"}:
                self.loss_functions[task_name] = nn.BCEWithLogitsLoss(reduction="none")

            elif config.task_type == "regression":
                self.loss_functions[task_name] = nn.MSELoss()

        logger.info("MultiTaskLoss initialized with tasks: %s", list(task_configs.keys()))

    
    def get_task_configs(self) -> Dict[str, TaskLossConfig]:
        """
        Return task configuration dictionary.
        """
        return self.task_configs

    @classmethod
    def from_task_settings(
        cls,
        task_settings: Dict[str, Dict[str, str | float]],
    ) -> "MultiTaskLoss":

        if not isinstance(task_settings, dict) or not task_settings:
            raise ValueError("task_settings must be a non-empty dictionary")

        configs: Dict[str, TaskLossConfig] = {}

        for task_name, settings in task_settings.items():

            if not isinstance(settings, dict):
                raise ValueError(f"{task_name}: settings must be dict")

            task_type = settings.get("task_type")
            if not isinstance(task_type, str):
                raise ValueError(f"{task_name}: missing/invalid task_type")

            weight_raw = settings.get("weight", 1.0)
            if not isinstance(weight_raw, (int, float)):
                raise ValueError(f"{task_name}: weight must be numeric")

            ignore_index = settings.get("ignore_index", -100)
            if not isinstance(ignore_index, int):
                raise ValueError(f"{task_name}: ignore_index must be int")

            configs[task_name] = TaskLossConfig(
                task_type=task_type,
                weight=float(weight_raw),
                ignore_index=ignore_index,
            )

        if not configs:
            raise ValueError("No valid task loss settings found")

        return cls(configs)

    def forward(
        self,
        logits: Dict[str, torch.Tensor],
        labels: Dict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:

        if not isinstance(logits, dict):
            raise TypeError("logits must be a dictionary")

        if not isinstance(labels, dict):
            raise TypeError("labels must be a dictionary")

        missing_logits = [t for t in self.task_configs if t not in logits]
        missing_labels = [t for t in self.task_configs if t not in labels]

        if missing_logits or missing_labels:
            msg = (
                f"MultiTaskLoss: missing_logits={missing_logits}, "
                f"missing_labels={missing_labels}"
            )
            if self.strict:
                raise RuntimeError(msg)
            logger.warning(msg)

        task_losses: Dict[str, torch.Tensor] = {}
        total_loss: Optional[torch.Tensor] = None

        for task_name, task_config in self.task_configs.items():

            if task_name not in logits or task_name not in labels:
                continue

            task_logits = logits[task_name]
            task_labels = labels[task_name]

            if task_logits.numel() == 0:
                raise RuntimeError(f"Empty logits for task {task_name}")

            if task_labels.device != task_logits.device:
                task_labels = task_labels.to(task_logits.device)

            # AMP stability
            task_logits = task_logits.float()

            if task_logits.shape[0] != task_labels.shape[0]:
                raise ValueError(
                    f"Batch mismatch for {task_name}: "
                    f"{task_logits.shape} vs {task_labels.shape}"
                )

            loss_fn = self.loss_functions[task_name]
            task_type = task_config.task_type

            # dtype normalization
            if task_type in {"binary", "multi_label", "regression"}:
                task_labels = task_labels.to(torch.float32)
            elif task_type == "multi_class":
                task_labels = task_labels.to(torch.long)

            # ---- task logic ----

            if task_type == "multi_class":

                if task_labels.dim() == 2:
                    task_labels = task_labels.argmax(dim=1)

                loss = loss_fn(task_logits, task_labels)

            elif task_type == "binary":

                if task_logits.dim() == 1:
                    task_logits = task_logits.unsqueeze(1)

                if task_labels.dim() == 1:
                    task_labels = task_labels.unsqueeze(1)

                loss = loss_fn(task_logits, task_labels)

            elif task_type == "multi_label":

                if task_logits.shape != task_labels.shape:
                    raise ValueError(
                        f"Shape mismatch for {task_name}: "
                        f"{task_logits.shape} vs {task_labels.shape}"
                    )

                ignore_index = float(task_config.ignore_index)
                valid_mask = task_labels.ne(ignore_index)
                if not bool(valid_mask.any()):
                    continue

                safe_labels = torch.where(valid_mask, task_labels, torch.zeros_like(task_labels))
                raw_loss = loss_fn(task_logits, safe_labels)
                masked_loss = raw_loss * valid_mask.to(raw_loss.dtype)
                loss = masked_loss.sum() / valid_mask.sum().clamp_min(1).to(raw_loss.dtype)

            elif task_type == "regression":

                task_labels = task_labels.view_as(task_logits)
                loss = loss_fn(task_logits, task_labels)

            else:
                raise RuntimeError(f"Unsupported task type: {task_type}")

            loss = loss.mean()

            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite loss detected in task {task_name}")

            weight = float(task_config.weight)
            if weight < 0:
                raise ValueError(f"Invalid weight for {task_name}: {weight}")

            weighted_loss = loss * weight

            task_losses[task_name] = weighted_loss

            total_loss = (
                weighted_loss if total_loss is None else total_loss + weighted_loss
            )

        if total_loss is None:
            raise RuntimeError("No task losses were computed")

        return total_loss, task_losses
