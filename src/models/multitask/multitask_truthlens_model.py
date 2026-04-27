from __future__ import annotations

import logging
from typing import Dict, Any

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


# =========================================================
# MULTI-TASK TRUTHLENS MODEL
# =========================================================

class MultiTaskTruthLensModel(nn.Module):
    
    def __init__(
        self,
        encoder: nn.Module,
        task_heads: Dict[str, nn.Module],
    ):
        super().__init__()

        if not isinstance(task_heads, dict) or not task_heads:
            raise ValueError("task_heads must be non-empty dict")

        self.encoder = encoder
        self.task_heads = nn.ModuleDict(task_heads)

        logger.info(
            "MultiTaskTruthLensModel initialized | tasks=%s",
            list(task_heads.keys()),
        )

    # =====================================================
    # FORWARD
    # =====================================================

    def forward(self, **inputs: Any) -> Dict[str, Any]:
      
        # -------------------------
        # ENCODER
        # -------------------------
        encoder_outputs = self.encoder(**inputs)

        # HuggingFace compatibility
        if isinstance(encoder_outputs, dict):
            pooled = encoder_outputs.get("pooler_output")
            hidden = encoder_outputs.get("last_hidden_state")

        else:
            pooled = getattr(encoder_outputs, "pooler_output", None)
            hidden = getattr(encoder_outputs, "last_hidden_state", None)

        # fallback (important for models like RoBERTa)
        if pooled is None:
            if hidden is None:
                raise RuntimeError("Encoder did not return usable outputs")
            pooled = hidden[:, 0]  # CLS token

        # -------------------------
        # TASK HEADS
        # -------------------------
        task_logits: Dict[str, torch.Tensor] = {}

        for task_name, head in self.task_heads.items():
            try:
                logits = head(pooled)
            except Exception as e:
                raise RuntimeError(
                    f"Head '{task_name}' forward failed: {e}"
                ) from e

            if not torch.is_tensor(logits):
                raise TypeError(f"{task_name}: head must return Tensor")

            task_logits[task_name] = logits

        # -------------------------
        # OUTPUT
        # -------------------------
        return {
            "task_logits": task_logits
        }

    # =====================================================
    # UTILITIES
    # =====================================================

    def get_task_names(self):
        return list(self.task_heads.keys())

    def freeze_encoder(self):
        for p in self.encoder.parameters():
            p.requires_grad = False
        logger.info("Encoder frozen")

    def unfreeze_encoder(self):
        for p in self.encoder.parameters():
            p.requires_grad = True
        logger.info("Encoder unfrozen")

    def freeze_heads(self):
        for head in self.task_heads.values():
            for p in head.parameters():
                p.requires_grad = False
        logger.info("All task heads frozen")

    def unfreeze_heads(self):
        for head in self.task_heads.values():
            for p in head.parameters():
                p.requires_grad = True
        logger.info("All task heads unfrozen")

    def freeze_task(self, task_name: str):
        if task_name not in self.task_heads:
            raise ValueError(f"Unknown task: {task_name}")

        for p in self.task_heads[task_name].parameters():
            p.requires_grad = False

        logger.info("Task '%s' frozen", task_name)

    def unfreeze_task(self, task_name: str):
        if task_name not in self.task_heads:
            raise ValueError(f"Unknown task: {task_name}")

        for p in self.task_heads[task_name].parameters():
            p.requires_grad = True

        logger.info("Task '%s' unfrozen", task_name)

    def get_trainable_parameters(self):
        return [p for p in self.parameters() if p.requires_grad]

    def extra_repr(self) -> str:
        return f"tasks={list(self.task_heads.keys())}"

# Backward-compat alias: MultiTaskTruthLensConfig for callers expecting a Config name.
class MultiTaskTruthLensConfig:
    """Lightweight stub config; the actual model is configured via dict kwargs."""
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)

