from __future__ import annotations

import logging
from typing import Dict, Optional, Any

import torch
import torch.nn as nn


logger = logging.getLogger(__name__)


class MultiTaskHead(nn.Module):

    def __init__(self) -> None:
        super().__init__()

        self.task_heads: nn.ModuleDict = nn.ModuleDict()
        self.loss_fns: Dict[str, nn.Module] = {}
        self.task_weights: Dict[str, float] = {}

    # =====================================================
    # REGISTER
    # =====================================================

    def register_task(
        self,
        task_name: str,
        head: nn.Module,
        loss_fn: Optional[nn.Module] = None,
        weight: float = 1.0,
    ) -> None:

        if not isinstance(task_name, str) or not task_name.strip():
            raise ValueError("task_name must be a valid string")

        if task_name in self.task_heads:
            raise ValueError(f"Task '{task_name}' already registered")

        if not isinstance(head, nn.Module):
            raise TypeError("head must be nn.Module")

        self.task_heads[task_name] = head

        if loss_fn is not None:
            self.loss_fns[task_name] = loss_fn

        self.task_weights[task_name] = float(weight)

        logger.info("Registered multitask head: %s", task_name)

    # =====================================================
    # FORWARD
    # =====================================================

    def forward(
        self,
        features: torch.Tensor,
        labels: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Dict[str, Any]:

        if features is None:
            raise ValueError("features cannot be None")

        if features.dim() != 2:
            raise ValueError(f"Expected 2D tensor, got {features.shape}")

        if not features.is_contiguous():
            features = features.contiguous()

        outputs: Dict[str, Any] = {
            "tasks": {},
        }

        total_loss: Optional[torch.Tensor] = None

        for task_name, head in self.task_heads.items():

            head_output = head(features)

            if not isinstance(head_output, dict):
                logits = head_output
                task_output = {"logits": logits}
            else:
                task_output = head_output

                if "logits" not in task_output:
                    raise RuntimeError(
                        f"Task '{task_name}' must return logits"
                    )

                logits = task_output["logits"]

            outputs["tasks"][task_name] = task_output

            # -------------------------
            # LOSS
            # -------------------------
            if labels is not None and task_name in labels:

                if task_name not in self.loss_fns:
                    raise RuntimeError(
                        f"No loss function for task '{task_name}'"
                    )

                loss_fn = self.loss_fns[task_name]
                task_labels = labels[task_name]

                loss = loss_fn(logits, task_labels)

                weight = self.task_weights.get(task_name, 1.0)

                weighted_loss = weight * loss

                task_output["loss"] = loss
                task_output["weighted_loss"] = weighted_loss

                if total_loss is None:
                    total_loss = weighted_loss
                else:
                    total_loss = total_loss + weighted_loss

        if total_loss is not None:
            outputs["total_loss"] = total_loss

        return outputs

    # =====================================================
    # PREDICT
    # =====================================================

    @torch.no_grad()
    def predict(self, features: torch.Tensor) -> Dict[str, Any]:

        was_training = self.training
        self.eval()

        try:
            outputs = self.forward(features)
        finally:
            if was_training:
                self.train()

        predictions: Dict[str, Any] = {}

        for task_name, task_output in outputs["tasks"].items():

            predictions[task_name] = {
                "predictions": task_output.get("predictions"),
                "probabilities": task_output.get("probabilities"),
                "confidence": task_output.get("confidence"),
            }

        return predictions

    # =====================================================
    # UTILS
    # =====================================================

    def get_tasks(self) -> Dict[str, nn.Module]:
        return dict(self.task_heads)

    def set_task_weight(self, task_name: str, weight: float) -> None:
        if task_name not in self.task_heads:
            raise ValueError(f"Task '{task_name}' not found")
        self.task_weights[task_name] = float(weight)

    def get_task_weights(self) -> Dict[str, float]:
        return dict(self.task_weights)