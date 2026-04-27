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

    def __init__(self, task_configs: Dict[str, Dict[str, Any]]) -> None:
        super().__init__()

        if not isinstance(task_configs, dict) or not task_configs:
            raise ValueError("task_configs must be non-empty dict")

        self.task_configs = task_configs
        self.task_heads: nn.ModuleDict = nn.ModuleDict()
        self.loss_functions: Dict[str, nn.Module] = {}

    # =====================================================
    # ENCODE
    # =====================================================

    @abstractmethod
    def encode(self, *inputs: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        raise NotImplementedError

    # =====================================================
    # REGISTER
    # =====================================================

    def register_task_head(
        self,
        task_name: str,
        head: nn.Module,
        loss_fn: nn.Module,
    ) -> None:

        if task_name in self.task_heads:
            raise ValueError(f"Task already exists: {task_name}")

        self.task_heads[task_name] = head
        self.loss_functions[task_name] = loss_fn

        logger.info("Registered head: %s", task_name)

    # =====================================================
    # FORWARD
    # =====================================================

    def forward(
        self,
        *inputs: torch.Tensor,
        labels: Optional[Dict[str, torch.Tensor]] = None,
        task: Optional[str] = None,
        return_features: bool = False,
        **kwargs: Any,
    ) -> Dict[str, Any]:

        shared = self.encode(*inputs, **kwargs)

        if shared.dim() != 2:
            raise ValueError(f"Expected 2D features, got {shared.shape}")

        active_task = task
        if active_task is None and labels and len(labels) == 1:
            active_task = next(iter(labels))

        task_list = [active_task] if active_task else list(self.task_heads.keys())

        outputs: Dict[str, Any] = {"tasks": {}}
        total_loss: Optional[torch.Tensor] = None

        for name in task_list:

            head = self.task_heads.get(name)
            if head is None:
                raise ValueError(f"No head: {name}")

            logits = head(shared)

            cfg = self.task_configs.get(name, {})
            task_type = cfg.get("type", "classification")

            task_out: Dict[str, torch.Tensor] = {"logits": logits}

            # Derived per-task statistics are inference-only; computing
            # them in training mode is wasted compute that also inflates
            # the autograd graph for tensors the loss never touches (P1).
            if not self.training:
                if task_type == "multilabel":
                    probs = torch.sigmoid(logits)
                    preds = (probs > 0.5).long()
                else:
                    probs = F.softmax(logits, dim=-1)
                    preds = torch.argmax(probs, dim=-1)

                confidence = probs.max(dim=-1).values
                entropy = -torch.sum(probs * torch.log(probs + 1e-12), dim=-1)

                task_out["probabilities"] = probs
                task_out["predictions"] = preds
                task_out["confidence"] = confidence
                task_out["entropy"] = entropy

            if labels and name in labels:

                loss_fn = self.loss_functions.get(name)
                if loss_fn is None:
                    raise RuntimeError(f"No loss_fn for {name}")

                target = labels[name]

                if task_type == "multilabel":
                    target = target.float()

                loss = loss_fn(logits, target)
                task_out["loss"] = loss

                total_loss = loss if active_task else (
                    loss if total_loss is None else total_loss + loss
                )

            outputs["tasks"][name] = task_out

        if active_task:
            outputs.update(outputs["tasks"][active_task])

        if total_loss is not None:
            outputs["loss"] = total_loss

        if return_features:
            outputs["shared_features"] = shared

        return outputs

    # =====================================================
    # PREDICT
    # =====================================================

    @torch.inference_mode()
    def predict(
        self,
        *inputs: torch.Tensor,
        task: Optional[str] = None,
        **kwargs: Any,
    ) -> Dict[str, Dict[str, torch.Tensor]]:

        was_training = self.training
        self.eval()

        try:
            outputs = self.forward(*inputs, task=task, **kwargs)
        finally:
            if was_training:
                self.train()

        return {
            name: {
                "predictions": out["predictions"],
                "probabilities": out["probabilities"],
                "confidence": out["confidence"],
            }
            for name, out in outputs["tasks"].items()
        }

    # =====================================================
    # TASKS
    # =====================================================

    def get_tasks(self) -> Dict[str, Dict[str, Any]]:
        return self.task_configs