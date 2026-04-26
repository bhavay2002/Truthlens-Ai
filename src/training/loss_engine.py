from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, Any, Optional, Tuple

import torch

from src.models.loss.multitask_loss import MultiTaskLoss, TaskLossConfig
from src.models.loss.base_balancer import BaseBalancer

logger = logging.getLogger(__name__)


# =========================================================
# CONFIG
# =========================================================

@dataclass
class LossEngineConfig:
    task_types: Dict[str, str]
    task_weights: Optional[Dict[str, float]] = None
    ignore_index: int = -100

    normalization: str = "active"
    use_normalizer: bool = True
    use_coverage: bool = True


# =========================================================
# LOSS ENGINE
# =========================================================

class LossEngine:

    def __init__(self, config: LossEngineConfig):

        self.config = config

        # -------------------------------------------------
        # BUILD TASK CONFIGS
        # -------------------------------------------------

        task_configs: Dict[str, TaskLossConfig] = {}

        for task, task_type in config.task_types.items():
            weight = (config.task_weights or {}).get(task, 1.0)

            task_configs[task] = TaskLossConfig(
                task_type=task_type,
                weight=float(weight),
                ignore_index=config.ignore_index,
            )

        # -------------------------------------------------
        # CORE LOSS MODULE
        # -------------------------------------------------

        self.loss_module = MultiTaskLoss(
            task_configs=task_configs,
            normalization=config.normalization,
            use_normalizer=config.use_normalizer,
            use_coverage=config.use_coverage,
        )

        self._balancer: Optional[BaseBalancer] = None

        logger.info(
            "LossEngine initialized | tasks=%s | norm=%s",
            list(task_configs.keys()),
            config.normalization,
        )

    # =====================================================
    # BALANCER
    # =====================================================

    def attach_balancer(self, balancer: BaseBalancer) -> None:

        if not isinstance(balancer, BaseBalancer):
            raise TypeError("balancer must inherit from BaseBalancer")

        self._balancer = balancer
        self.loss_module.attach_task_balancer(balancer)

        logger.info("Balancer attached: %s", balancer.__class__.__name__)

    # =====================================================
    # MAIN
    # =====================================================

    def compute(
        self,
        outputs: Dict[str, Any],
        batch: Dict[str, Any],
        *,
        shared_parameters=None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:

        if "task_logits" not in outputs:
            raise RuntimeError("Missing 'task_logits'")

        if "labels" not in batch:
            raise RuntimeError("Missing 'labels'")

        logits = outputs["task_logits"]
        labels = batch["labels"]

        # -------------------------------------------------
        # CORE LOSS
        # -------------------------------------------------

        total_loss, task_losses = self.loss_module(logits, labels)

        # -------------------------------------------------
        # NUMERICAL SAFETY
        # -------------------------------------------------

        if not torch.isfinite(total_loss):
            raise RuntimeError(f"Non-finite total loss: {total_loss.item()}")

        for task, loss in task_losses.items():
            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite loss in task '{task}'")

        # -------------------------------------------------
        # BALANCER PRE-BACKWARD
        # -------------------------------------------------

        if self._balancer is not None:
            try:
                self._balancer.on_before_backward(
                    task_losses={k: v.detach() for k, v in task_losses.items()},
                    shared_parameters=shared_parameters,
                )
            except Exception as e:
                logger.warning("Balancer pre-backward failed: %s", e)

        # -------------------------------------------------
        # DEBUG ATTACHMENTS (SAFE)
        # -------------------------------------------------

        outputs["task_losses"] = {
            k: v.detach() for k, v in task_losses.items()
        }

        outputs["total_loss"] = total_loss.detach()

        outputs["loss_stats"] = {
            "num_tasks": len(task_losses),
            "mean_loss": float(
                torch.stack(list(task_losses.values())).mean().item()
            )
        }

        return total_loss, task_losses

    # =====================================================
    # TRAINING HOOKS
    # =====================================================

    def on_after_backward(self) -> None:

        if self._balancer is not None:
            try:
                self._balancer.on_after_backward()
            except Exception as e:
                logger.warning("Balancer post-backward failed: %s", e)

    def on_step_end(self) -> None:

        if self._balancer is not None:
            try:
                self._balancer.on_step_end()
            except Exception as e:
                logger.warning("Balancer step-end failed: %s", e)

    # =====================================================
    # STATS
    # =====================================================

    def get_stats(self) -> Dict[str, Any]:

        stats = {}

        if hasattr(self.loss_module, "get_stats"):
            stats.update(self.loss_module.get_stats())

        if self._balancer is not None:
            if hasattr(self._balancer, "get_weights"):
                stats["balancer_weights"] = self._balancer.get_weights()

        return stats

    # =====================================================
    # RESET
    # =====================================================

    def reset(self) -> None:

        if hasattr(self.loss_module, "reset"):
            self.loss_module.reset()

        logger.info("LossEngine reset complete")