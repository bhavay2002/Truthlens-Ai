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
    # Optional positive-class weight for BCE / multi-label tasks. Either a
    # scalar (broadcast across all label dimensions) or a 1-D tensor with
    # one entry per label class. ``None`` disables class re-weighting.
    pos_weight: Optional[torch.Tensor] = None


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
                # pos_weight is registered as a (non-trainable) buffer so it
                # moves with .to(device) and survives state_dict round-trips.
                bce = nn.BCEWithLogitsLoss(
                    reduction="none",
                    pos_weight=config.pos_weight if config.pos_weight is not None else None,
                )
                self.loss_functions[task_name] = bce

            elif config.task_type == "regression":
                self.loss_functions[task_name] = nn.MSELoss()

        # ---- Optional Kendall-uncertainty task balancer (Phase-4 of the
        # playbook). When attached via ``attach_task_balancer``, the per-task
        # weighted losses are *re-combined* through the balancer's
        # learnable log-variances instead of a simple sum. The balancer's
        # parameters must be added to the optimizer by the caller (it is a
        # sub-module so ``model.parameters()`` will pick it up automatically
        # when attached before optimizer construction).
        self.task_balancer: Optional[nn.Module] = None

        # ---- EMA-based task coverage tracker (Phase-3 of the multi-task
        # stabilization playbook). Tracks, per task, the smoothed probability
        # that the task has any valid label in a batch. The inverse is used
        # as a multiplier so rare tasks get boosted without re-tuning the
        # static `weight` field. Opt-in via ``ema_weighting=True``.
        self._coverage_ema: Dict[str, float] = {name: 0.0 for name in task_configs}
        self._coverage_steps: Dict[str, int] = {name: 0 for name in task_configs}
        self._ema_alpha: float = 0.1
        self.ema_weighting: bool = False
        # Floor stops the inverse-EMA multiplier from blowing up early in
        # training when a task has been observed only a handful of times.
        self._ema_floor: float = 0.05
        # Cap so a single near-zero task doesn't overwhelm the loss.
        self._ema_cap: float = 10.0

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

    def attach_task_balancer(self, balancer: nn.Module) -> None:
        """Attach a Kendall-uncertainty TaskBalancer (or compatible module).

        The balancer must implement ``forward(task_losses: Dict[str, Tensor])
        -> Tensor``. After attachment, ``MultiTaskLoss.forward`` returns the
        balancer-combined total instead of the simple weighted sum. The
        balancer is registered as a sub-module so its parameters appear in
        ``model.parameters()`` automatically.
        """
        if not hasattr(balancer, "forward"):
            raise TypeError("balancer must be an nn.Module with forward()")
        self.task_balancer = balancer
        logger.info(
            "MultiTaskLoss: TaskBalancer ATTACHED (%s)",
            balancer.__class__.__name__,
        )

    def enable_ema_weighting(
        self,
        *,
        alpha: float = 0.1,
        floor: float = 0.05,
        cap: float = 10.0,
    ) -> None:
        """Turn on EMA-coverage based dynamic task weighting.

        Multiplies each task's static weight by ``min(1/cov_ema, cap)`` where
        ``cov_ema`` is the smoothed per-batch probability that the task has
        any valid label. Rare tasks therefore get gradient-weight boosts
        proportional to how often they go un-supervised.
        """
        if not 0.0 < alpha <= 1.0:
            raise ValueError(f"alpha must be in (0,1], got {alpha}")
        if floor <= 0.0:
            raise ValueError(f"floor must be positive, got {floor}")
        if cap < 1.0:
            raise ValueError(f"cap must be >= 1.0, got {cap}")
        self.ema_weighting = True
        self._ema_alpha = float(alpha)
        self._ema_floor = float(floor)
        self._ema_cap = float(cap)
        logger.info(
            "MultiTaskLoss EMA-coverage weighting ENABLED (alpha=%.3f floor=%.3f cap=%.2f)",
            self._ema_alpha, self._ema_floor, self._ema_cap,
        )

    def coverage_report(self) -> Dict[str, float]:
        """Return the current per-task EMA coverage estimate."""
        return dict(self._coverage_ema)

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

            pos_weight_raw = settings.get("pos_weight")
            pos_weight_tensor: Optional[torch.Tensor] = None
            if pos_weight_raw is not None:
                if torch.is_tensor(pos_weight_raw):
                    pos_weight_tensor = pos_weight_raw.float()
                else:
                    try:
                        pos_weight_tensor = torch.as_tensor(
                            pos_weight_raw, dtype=torch.float32
                        )
                    except Exception as exc:
                        raise ValueError(
                            f"{task_name}: pos_weight must be tensor-convertible ({exc})"
                        ) from exc

            configs[task_name] = TaskLossConfig(
                task_type=task_type,
                weight=float(weight_raw),
                ignore_index=ignore_index,
                pos_weight=pos_weight_tensor,
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
        # Raw (pre-weight, pre-EMA) per-task losses fed to the optional
        # Kendall TaskBalancer when attached. Always allocated so the
        # balancer hook stays simple.
        raw_losses_for_balancer: Dict[str, torch.Tensor] = {}
        total_loss: Optional[torch.Tensor] = None
        active_heads = 0

        # Bump the global call counter once per forward(); per-head increments
        # happen below only when a head actually contributes a loss.
        self.total_forward_calls += 1

        # ---- Per-task coverage probe (used both for the EMA weighting
        # multiplier below and for the empty-batch guard at the bottom).
        per_task_coverage: Dict[str, bool] = {}
        for _name, _cfg in self.task_configs.items():
            _lbl = labels.get(_name)
            if not torch.is_tensor(_lbl) or _lbl.numel() == 0:
                per_task_coverage[_name] = False
                continue
            if _cfg.task_type == "multi_class":
                per_task_coverage[_name] = bool(_lbl.ne(_cfg.ignore_index).any())
            elif _cfg.task_type in {"binary", "multi_label"}:
                per_task_coverage[_name] = bool(_lbl.ne(float(_cfg.ignore_index)).any())
            else:
                per_task_coverage[_name] = True
            self._coverage_steps[_name] += 1
            self._coverage_ema[_name] = (
                self._ema_alpha * (1.0 if per_task_coverage[_name] else 0.0)
                + (1.0 - self._ema_alpha) * self._coverage_ema[_name]
            )

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

            # EMA-coverage multiplier (Phase-3 of the playbook): rare tasks
            # get inversely boosted so they accumulate gradient signal at
            # roughly the same long-horizon rate as well-supervised tasks.
            if self.ema_weighting:
                cov = max(self._coverage_ema.get(task_name, 0.0), self._ema_floor)
                ema_mul = min(1.0 / cov, self._ema_cap)
                weight = weight * ema_mul

            weighted_loss = loss * weight

            task_losses[task_name] = weighted_loss
            active_heads += 1
            self.head_call_counts[task_name] += 1
            # `loss` is the (pre-weight, pre-EMA) per-task scalar — exactly
            # what the Kendall-uncertainty balancer wants to combine.
            if self.task_balancer is not None:
                raw_losses_for_balancer[task_name] = loss

            total_loss = (
                weighted_loss if total_loss is None else total_loss + weighted_loss
            )

        if total_loss is None or active_heads == 0:
            # The strict path raises so a chronically-misconfigured sampler
            # (every batch fully masked) cannot be silently swallowed. The
            # opt-in soft path (TRUTHLENS_SKIP_EMPTY_BATCH=1) returns a
            # zero scalar carrying gradient through `0 * sum(logits)` so the
            # caller can detect-and-skip without the autograd graph blowing
            # up. The trainer treats `last_active_heads == 0` as "skip
            # optimizer step" so no parameters are nudged by the zero loss.
            import os as _os
            if _os.environ.get("TRUTHLENS_SKIP_EMPTY_BATCH", "0") == "1":
                _grad_carrier = None
                for _name, _t in logits.items():
                    if torch.is_tensor(_t) and _t.requires_grad:
                        _zero = (_t.float().sum() * 0.0)
                        _grad_carrier = (
                            _zero if _grad_carrier is None else _grad_carrier + _zero
                        )
                if _grad_carrier is None:
                    _grad_carrier = torch.zeros((), requires_grad=False)
                logger.warning(
                    "MultiTaskLoss: every head masked out this batch — "
                    "returning zero loss (skip-empty-batch mode). Check "
                    "sampling if this fires repeatedly."
                )
                self.last_active_heads = 0
                return _grad_carrier, {}
            raise RuntimeError(
                "No task losses were computed — every head was masked out for "
                "this batch. Check label sparsity / batch sampling. Set "
                "TRUTHLENS_SKIP_EMPTY_BATCH=1 to soft-skip instead."
            )

        # ---- Kendall-uncertainty re-combination (Phase-4). Replaces the
        # naive weighted sum with ``sum_t precision_t * loss_t + log_var_t``
        # so the model learns its own per-task weights jointly with the
        # rest of the parameters.
        if self.task_balancer is not None and raw_losses_for_balancer:
            try:
                total_loss = self.task_balancer(raw_losses_for_balancer)
            except Exception as exc:
                logger.warning(
                    "TaskBalancer failed (%s) — falling back to weighted-sum total.",
                    exc,
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
