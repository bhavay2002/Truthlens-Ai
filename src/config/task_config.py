"""
File: task_config.py

Central task registry for TruthLens multi-task system.
Defines task types, output dimensions, losses, thresholds,
and evaluation behavior.
"""

from __future__ import annotations

from typing import Dict, Any


# =========================================================
# TASK CONFIG (SINGLE SOURCE OF TRUTH)
# =========================================================
TASK_CONFIG: Dict[str, Dict[str, Any]] = {

    # -----------------------------------------------------
    # BIAS DETECTION
    # -----------------------------------------------------
    "bias": {
        "type": "binary",
        "num_classes": 2,
        "loss": "bce",
        "loss_weight": 1.0,
        "threshold": 0.5,
    },

    # -----------------------------------------------------
    # IDEOLOGY CLASSIFICATION
    # -----------------------------------------------------
    "ideology": {
        "type": "multiclass",
        "num_classes": 3,
        "loss": "cross_entropy",
        "loss_weight": 1.0,
    },

    # -----------------------------------------------------
    # PROPAGANDA DETECTION
    # -----------------------------------------------------
    "propaganda": {
        "type": "binary",
        "num_classes": 2,
        "loss": "bce",
        "loss_weight": 1.0,
        "threshold": 0.5,
    },

    # -----------------------------------------------------
    # EMOTION (MULTILABEL)
    # -----------------------------------------------------
    "emotion": {
        "type": "multilabel",
        "num_labels": 20,
        "loss": "bce",
        "loss_weight": 2.0,
        "threshold": 0.5,
        "auto_threshold": True,  # enables tuning
    },

    # -----------------------------------------------------
    # SENTIMENT
    # -----------------------------------------------------
    "sentiment": {
        "type": "multiclass",
        "num_classes": 3,
        "loss": "cross_entropy",
        "loss_weight": 1.0,
    },

    # -----------------------------------------------------
    # NARRATIVE FRAME (MULTILABEL)
    # -----------------------------------------------------
    "narrative_frame": {
        "type": "multilabel",
        "num_labels": 10,
        "loss": "bce",
        "loss_weight": 2.0,
        "threshold": 0.5,
    },
}


# =========================================================
# VALIDATION (CRITICAL)
# =========================================================
def validate_task_config():

    for task, cfg in TASK_CONFIG.items():

        if "type" not in cfg:
            raise ValueError(f"{task}: missing type")

        if cfg["type"] == "multiclass" and "num_classes" not in cfg:
            raise ValueError(f"{task}: num_classes required")

        if cfg["type"] == "multilabel" and "num_labels" not in cfg:
            raise ValueError(f"{task}: num_labels required")

        if "loss" not in cfg:
            raise ValueError(f"{task}: missing loss")

        if "loss_weight" not in cfg:
            raise ValueError(f"{task}: missing loss_weight")


# Run validation on import
validate_task_config()


# =========================================================
# HELPERS (USED EVERYWHERE)
# =========================================================
def get_task_type(task: str) -> str:
    return TASK_CONFIG[task]["type"]


def get_output_dim(task: str) -> int:
    cfg = TASK_CONFIG[task]

    if cfg["type"] == "multilabel":
        return cfg["num_labels"]

    return cfg.get("num_classes", 1)


def get_loss_name(task: str) -> str:
    return TASK_CONFIG[task]["loss"]


def get_loss_weight(task: str) -> float:
    return float(TASK_CONFIG[task].get("loss_weight", 1.0))


def get_threshold(task: str) -> float:
    return float(TASK_CONFIG[task].get("threshold", 0.5))


def use_auto_threshold(task: str) -> bool:
    return bool(TASK_CONFIG[task].get("auto_threshold", False))


def is_multilabel(task: str) -> bool:
    return TASK_CONFIG[task]["type"] == "multilabel"


def is_binary(task: str) -> bool:
    return TASK_CONFIG[task]["type"] == "binary"


def is_multiclass(task: str) -> bool:
    return TASK_CONFIG[task]["type"] == "multiclass"