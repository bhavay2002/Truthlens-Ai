from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional, Any

import torch


# =========================================================
# TASK TYPES
# =========================================================

_MULTICLASS_TASKS = {"bias", "ideology", "propaganda"}
_MULTILABEL_TASKS = {"narrative", "narrative_frame", "emotion"}
_VALID_TASKS = _MULTICLASS_TASKS | _MULTILABEL_TASKS


# =========================================================
# UTILS
# =========================================================

def _validate_task(task: str):
    if task not in _VALID_TASKS:
        raise ValueError(f"Unknown task: {task}")


def _compute_confidence(
    task: str,
    probabilities: Optional[torch.Tensor],
) -> Optional[torch.Tensor]:

    if probabilities is None:
        return None

    if task in _MULTICLASS_TASKS:
        return probabilities.max(dim=-1).values

    if task in _MULTILABEL_TASKS:
        return probabilities.mean(dim=-1)

    return None


def _compute_entropy(probs: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
    if probs is None:
        return None
    return -torch.sum(probs * torch.log(probs + 1e-12), dim=-1)


# =========================================================
# TASK OUTPUT
# =========================================================

@dataclass
class TaskPrediction:

    logits: Optional[torch.Tensor] = None
    probabilities: Optional[torch.Tensor] = None
    predictions: Optional[torch.Tensor] = None
    confidence: Optional[torch.Tensor] = None
    entropy: Optional[torch.Tensor] = None
    metadata: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "logits": self.logits,
            "probabilities": self.probabilities,
            "predictions": self.predictions,
            "confidence": self.confidence,
            "entropy": self.entropy,
            "metadata": self.metadata,
        }


# =========================================================
# MAIN OUTPUT
# =========================================================

@dataclass
class PredictionOutput:

    tasks: Dict[str, TaskPrediction] = field(default_factory=dict)
    metadata: Optional[Dict[str, Any]] = None

    # =====================================================
    # BUILDERS
    # =====================================================

    @classmethod
    def from_flat(
        cls,
        outputs: Dict[str, Any],
        *,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> "PredictionOutput":

        if not isinstance(outputs, dict):
            raise TypeError("outputs must be dict")

        obj = cls(metadata=metadata)

        if "tasks" in outputs:
            for name, val in outputs["tasks"].items():
                obj.add_task(
                    name,
                    logits=val.get("logits"),
                    probabilities=val.get("probabilities"),
                    predictions=val.get("predictions"),
                    confidence=val.get("confidence"),
                    metadata=val.get("metadata"),
                )
            return obj

        for k, v in outputs.items():
            if "_" not in k:
                continue
            task, field = k.rsplit("_", 1)

            if task not in obj.tasks:
                obj.tasks[task] = TaskPrediction()

            if hasattr(obj.tasks[task], field):
                setattr(obj.tasks[task], field, v)

        return obj

    @classmethod
    def from_single_task(
        cls,
        task: str,
        outputs: Dict[str, Any],
        *,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> "PredictionOutput":

        _validate_task(task)

        obj = cls(metadata=metadata)

        probs = outputs.get("probabilities")

        obj.add_task(
            task,
            logits=outputs.get("logits"),
            probabilities=probs,
            predictions=outputs.get("predictions"),
            confidence=outputs.get("confidence")
            or _compute_confidence(task, probs),
        )

        return obj

    # =====================================================
    # ADD
    # =====================================================

    def add_task(
        self,
        task_name: str,
        logits: Optional[torch.Tensor] = None,
        probabilities: Optional[torch.Tensor] = None,
        predictions: Optional[torch.Tensor] = None,
        confidence: Optional[torch.Tensor] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:

        if confidence is None and probabilities is not None:
            try:
                confidence = _compute_confidence(task_name, probabilities)
            except Exception:
                pass

        entropy = _compute_entropy(probabilities)

        self.tasks[task_name] = TaskPrediction(
            logits=logits,
            probabilities=probabilities,
            predictions=predictions,
            confidence=confidence,
            entropy=entropy,
            metadata=metadata,
        )

    # =====================================================
    # ACCESS
    # =====================================================

    def get_task(self, task_name: str) -> TaskPrediction:
        if task_name not in self.tasks:
            raise KeyError(task_name)
        return self.tasks[task_name]

    # =====================================================
    # EXPORT
    # =====================================================

    def to_dict(self) -> Dict[str, Any]:

        return {
            "tasks": {
                k: v.to_dict()
                for k, v in self.tasks.items()
            },
            "metadata": self.metadata,
        }

    def to_lightweight(self) -> Dict[str, Any]:

        return {
            name: task.predictions
            for name, task in self.tasks.items()
        }