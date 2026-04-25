from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)

EPS = 1e-12


# =========================================================
# CONFIG
# =========================================================

@dataclass
class DistillationConfig:
    temperature: float = 2.0
    alpha: float = 0.5  # balance between hard + soft loss

    # feature distillation
    enable_feature_distillation: bool = False
    feature_weight: float = 0.5

    # attention distillation
    enable_attention_distillation: bool = False
    attention_weight: float = 0.5


# =========================================================
# CORE DISTILLATION LOSS
# =========================================================

class KnowledgeDistillationLoss(nn.Module):
    """
    Standard KD Loss:
        L = alpha * CE + (1 - alpha) * KL(student || teacher)
    """

    def __init__(self, config: DistillationConfig) -> None:
        super().__init__()
        self.config = config

    def forward(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        labels: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:

        T = self.config.temperature
        alpha = self.config.alpha

        # -------------------------
        # HARD LOSS (ground truth)
        # -------------------------
        hard_loss = F.cross_entropy(student_logits, labels)

        # -------------------------
        # SOFT LOSS (teacher)
        # -------------------------
        student_log_probs = F.log_softmax(student_logits / T, dim=-1)
        teacher_probs = F.softmax(teacher_logits / T, dim=-1)

        soft_loss = F.kl_div(
            student_log_probs,
            teacher_probs,
            reduction="batchmean",
        ) * (T * T)

        # -------------------------
        # TOTAL
        # -------------------------
        loss = alpha * hard_loss + (1 - alpha) * soft_loss

        return {
            "loss": loss,
            "hard_loss": hard_loss,
            "soft_loss": soft_loss,
        }


# =========================================================
# FEATURE DISTILLATION
# =========================================================

class FeatureDistillationLoss(nn.Module):
    """
    Align intermediate representations.
    """

    def __init__(self) -> None:
        super().__init__()

    def forward(
        self,
        student_features: torch.Tensor,
        teacher_features: torch.Tensor,
    ) -> torch.Tensor:

        return F.mse_loss(student_features, teacher_features)


# =========================================================
# ATTENTION DISTILLATION
# =========================================================

class AttentionDistillationLoss(nn.Module):
    """
    Align attention maps.
    """

    def __init__(self) -> None:
        super().__init__()

    def forward(
        self,
        student_attn: torch.Tensor,
        teacher_attn: torch.Tensor,
    ) -> torch.Tensor:

        # normalize
        student_attn = student_attn / (student_attn.sum(dim=-1, keepdim=True) + EPS)
        teacher_attn = teacher_attn / (teacher_attn.sum(dim=-1, keepdim=True) + EPS)

        return F.mse_loss(student_attn, teacher_attn)


# =========================================================
# FULL DISTILLATION MODULE
# =========================================================

class DistillationTrainer:
    """
    Combines:
        - KD loss
        - feature distillation
        - attention distillation
    """

    def __init__(
        self,
        config: DistillationConfig,
    ) -> None:

        self.config = config

        self.kd_loss = KnowledgeDistillationLoss(config)

        self.feature_loss = FeatureDistillationLoss()
        self.attn_loss = AttentionDistillationLoss()

    def compute_loss(
        self,
        student_outputs: Dict[str, Any],
        teacher_outputs: Dict[str, Any],
        labels: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:

        student_logits = student_outputs["logits"]
        teacher_logits = teacher_outputs["logits"]

        losses = self.kd_loss(
            student_logits=student_logits,
            teacher_logits=teacher_logits,
            labels=labels,
        )

        total_loss = losses["loss"]

        # -------------------------
        # FEATURE DISTILLATION
        # -------------------------
        if self.config.enable_feature_distillation:
            if "features" in student_outputs and "features" in teacher_outputs:
                feat_loss = self.feature_loss(
                    student_outputs["features"],
                    teacher_outputs["features"],
                )
                total_loss = total_loss + self.config.feature_weight * feat_loss
                losses["feature_loss"] = feat_loss

        # -------------------------
        # ATTENTION DISTILLATION
        # -------------------------
        if self.config.enable_attention_distillation:
            if "attentions" in student_outputs and "attentions" in teacher_outputs:
                attn_loss = self.attn_loss(
                    student_outputs["attentions"],
                    teacher_outputs["attentions"],
                )
                total_loss = total_loss + self.config.attention_weight * attn_loss
                losses["attention_loss"] = attn_loss

        losses["total_loss"] = total_loss

        return losses