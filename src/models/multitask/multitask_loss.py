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

    VALID_NORMALIZATIONS = {"active", "fixed", "sum"}

    def __init__(
        self,
        task_configs: Dict[str, TaskLossConfig],
        *,
        strict: bool = True,
        normalization: str = "active",
    ) -> None:
        super().__init__()

        if not isinstance(task_configs, dict) or not task_configs:
            raise ValueError("task_configs must be a non-empty dictionary")

        if normalization not in self.VALID_NORMALIZATIONS:
            raise ValueError(
                f"normalization must be one of {self.VALID_NORMALIZATIONS}, "
                f"got {normalization!r}"
            )

        self.task_configs = task_configs
        self.strict = strict
        self.normalization = normalization

        # Head-starvation tracker: counts how many forward() calls actually
        # produced a contributing loss for each task. Sparse-supervision
        # multi-task setups can silently starve a head when most of its
        # labels are missing, so we expose a counter the trainer can log.
        self.head_call_counts: Dict[str, int] = {name: 0 for name in task_configs}
        self.total_forward_calls: int = 0

        # Most recent forward()'s active-head count, exposed for trainer
        # logging without changing the (total_loss, task_losses) return shape.
        self.last_active_heads: int = 0

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

    def reset_head_stats(self) -> None:
        """Zero the per-head call counters (call at the start of an epoch)."""
        for name in self.head_call_counts:
            self.head_call_counts[name] = 0
        self.total_forward_calls = 0

    def head_starvation_report(self) -> Dict[str, float]:
        """Return per-head supervision rate over the current tracking window.

        Useful for logging: a value near 0.0 means the head almost never
        receives a labeled batch and is silently starved of gradient signal.
        """
        denom = max(self.total_forward_calls, 1)
        return {name: count / denom for name, count in self.head_call_counts.items()}

    @classmethod
    def from_task_settings(
        cls,
        task_settings: Dict[str, Dict[str, str | float]],
        *,
        normalization: str = "active",
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

        return cls(configs, normalization=normalization)

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
        active_heads = 0

        # Bump the global call counter once per forward(); per-head increments
        # happen below only when a head actually contributes a loss.
        self.total_forward_calls += 1

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

                valid_labels = task_labels.ne(task_config.ignore_index)
                if not bool(valid_labels.any()):
                    continue

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
            active_heads += 1
            self.head_call_counts[task_name] += 1

            total_loss = (
                weighted_loss if total_loss is None else total_loss + weighted_loss
            )

        if total_loss is None or active_heads == 0:
            raise RuntimeError(
                "No task losses were computed — every head was masked out for "
                "this batch. Check label sparsity / batch sampling."
            )

        # Normalize so gradient scale does not fluctuate with how many heads
        # happened to be supervised in a given batch.
        #   - "active": fair per-batch (divide by heads that contributed)
        #   - "fixed":  most stable gradients (divide by total configured heads)
        #   - "sum":    legacy behavior (no normalization)
        if self.normalization == "active":
            total_loss = total_loss / float(active_heads)
        elif self.normalization == "fixed":
            total_loss = total_loss / float(len(self.task_configs))
        # "sum" → leave as-is

        # Expose active-head count via a module attribute (no change to the
        # (total_loss, task_losses) return shape, so existing consumers that
        # iterate task_losses keep working unchanged).
        self.last_active_heads = active_heads

        return total_loss, task_losses
