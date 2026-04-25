from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional, Any

import torch
 

# =========================================================
# TASK OUTPUT
# =========================================================

@dataclass
class TaskOutput:

    logits: torch.Tensor
    probabilities: Optional[torch.Tensor] = None
    predictions: Optional[torch.Tensor] = None
    loss: Optional[torch.Tensor] = None

    # -------------------------
    # UTILITIES
    # -------------------------

    def detach(self) -> "TaskOutput":
        return TaskOutput(
            logits=self.logits.detach(),
            probabilities=self._safe_detach(self.probabilities),
            predictions=self._safe_detach(self.predictions),
            loss=self._safe_detach(self.loss),
        )

    def to(self, device: torch.device) -> "TaskOutput":
        return TaskOutput(
            logits=self.logits.to(device),
            probabilities=self._safe_to(self.probabilities, device),
            predictions=self._safe_to(self.predictions, device),
            loss=self._safe_to(self.loss, device),
        )

    def _safe_detach(self, x):
        return x.detach() if isinstance(x, torch.Tensor) else x

    def _safe_to(self, x, device):
        return x.to(device) if isinstance(x, torch.Tensor) else x


# =========================================================
# MULTI TASK OUTPUT
# =========================================================

@dataclass
class MultiTaskOutput:

    tasks: Dict[str, TaskOutput] = field(default_factory=dict)

    loss: Optional[torch.Tensor] = None
    task_losses: Optional[Dict[str, torch.Tensor]] = None

    metadata: Dict[str, Any] = field(default_factory=dict)

    # =====================================================
    # FACTORY
    # =====================================================

    @classmethod
    def from_model_outputs(cls, outputs: Dict[str, Any]) -> "MultiTaskOutput":

        if isinstance(outputs.get("multitask_output"), MultiTaskOutput):
            return outputs["multitask_output"]

        multitask = cls()

        # FAST PATH (preferred)
        if "task_logits" in outputs:
            for task, logits in outputs["task_logits"].items():
                multitask.tasks[task] = TaskOutput(logits=logits)

        # FALLBACK (legacy)
        else:
            for task_name, payload in outputs.items():

                if not isinstance(payload, dict):
                    continue

                logits = payload.get("logits")
                if not isinstance(logits, torch.Tensor):
                    continue

                multitask.tasks[task_name] = TaskOutput(
                    logits=logits,
                    probabilities=payload.get("probabilities"),
                    predictions=payload.get("predictions"),
                    loss=payload.get("loss"),
                )

        multitask.loss = outputs.get("loss")
        multitask.task_losses = outputs.get("task_losses")

        return multitask

    # =====================================================
    # LOSS ENGINE INTERFACE ( IMPORTANT)
    # =====================================================

    def to_loss_inputs(self) -> Dict[str, Dict[str, torch.Tensor]]:
        """
        Convert to LossEngine-compatible format.

        Returns:
            {
                "task_logits": {...}
            }
        """
        return {
            "task_logits": {
                task: out.logits for task, out in self.tasks.items()
            }
        }

    # =====================================================
    # ACCESSORS
    # =====================================================

    def get_logits(self, task_name: str) -> torch.Tensor:
        return self.tasks[task_name].logits

    def get_predictions(self, task_name: str):
        return self.tasks[task_name].predictions

    def get_probabilities(self, task_name: str):
        return self.tasks[task_name].probabilities

    def get_task_loss(self, task_name: str):
        return self.tasks[task_name].loss

    # =====================================================
    # DEVICE OPS
    # =====================================================

    def to(self, device: torch.device) -> "MultiTaskOutput":
        return MultiTaskOutput(
            tasks={k: v.to(device) for k, v in self.tasks.items()},
            loss=self.loss.to(device) if isinstance(self.loss, torch.Tensor) else self.loss,
            task_losses={
                k: v.to(device) for k, v in (self.task_losses or {}).items()
            } if self.task_losses else None,
            metadata=self.metadata,
        )

    def detach(self) -> "MultiTaskOutput":
        return MultiTaskOutput(
            tasks={k: v.detach() for k, v in self.tasks.items()},
            loss=self.loss.detach() if isinstance(self.loss, torch.Tensor) else self.loss,
            task_losses={
                k: v.detach() for k, v in (self.task_losses or {}).items()
            } if self.task_losses else None,
            metadata=self.metadata,
        )

    # =====================================================
    # SERIALIZATION
    # =====================================================

    def to_dict(self, detach: bool = True) -> Dict[str, Any]:

        result = {}

        for task_name, task_output in self.tasks.items():

            if detach:
                task_output = task_output.detach()

            result[task_name] = {
                "logits": task_output.logits,
                "probabilities": task_output.probabilities,
                "predictions": task_output.predictions,
                "loss": task_output.loss,
            }

        result["loss"] = self.loss.detach() if detach and isinstance(self.loss, torch.Tensor) else self.loss

        if self.task_losses:
            result["task_losses"] = {
                k: v.detach() if detach else v
                for k, v in self.task_losses.items()
            }

        if self.metadata:
            result["metadata"] = self.metadata

        return result

    def to_flat_prediction_dict(self) -> Dict[str, Any]:

        flat = {}

        for task_name, task_output in self.tasks.items():

            flat[f"{task_name}_logits"] = task_output.logits

            if task_output.probabilities is not None:
                flat[f"{task_name}_probabilities"] = task_output.probabilities

            if task_output.predictions is not None:
                flat[f"{task_name}_predictions"] = task_output.predictions

        if self.loss is not None:
            flat["loss"] = self.loss

        if self.task_losses:
            flat["task_losses"] = self.task_losses

        if self.metadata:
            flat["metadata"] = self.metadata

        return flat