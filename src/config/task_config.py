"""
File: task_config.py
Location: src/config/

Production-grade task registry for multi-task system.

This module:
- Builds task registry from YAML config
- Validates task definitions
- Provides fast lookup helpers
- Acts as SINGLE runtime source of truth

NOTE:
YAML config is the only source of truth.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict

from src.utils.config_loader import load_app_config

logger = logging.getLogger(__name__)


# =========================================================
# TASK DATACLASS (STRICT CONTRACT)
# =========================================================

@dataclass(slots=True, frozen=True)
class TaskDefinition:
    name: str
    task_type: str
    num_labels: int
    loss: str
    loss_weight: float
    threshold: float
    auto_threshold: bool


# =========================================================
# INTERNAL REGISTRY
# =========================================================

_TASK_REGISTRY: Dict[str, TaskDefinition] = {}


# =========================================================
# VALIDATION
# =========================================================

VALID_TASK_TYPES = {"binary", "multiclass", "multilabel"}
VALID_LOSSES = {"bce", "cross_entropy"}


def _validate_task(task: TaskDefinition):

    if task.task_type not in VALID_TASK_TYPES:
        raise ValueError(f"{task.name}: invalid task_type")

    if task.num_labels <= 0:
        raise ValueError(f"{task.name}: num_labels must be > 0")

    if task.loss not in VALID_LOSSES:
        raise ValueError(f"{task.name}: invalid loss")

    if task.loss_weight <= 0:
        raise ValueError(f"{task.name}: loss_weight must be > 0")

    if task.task_type == "multilabel" and task.threshold <= 0:
        raise ValueError(f"{task.name}: invalid threshold")


# =========================================================
# REGISTRY BUILDER
# =========================================================

def _build_registry():

    config = load_app_config()

    for task_name, task_cfg in config.tasks.items():

        # ---- derive defaults intelligently ----
        if task_cfg.task_type == "multilabel":
            loss = "bce"
        elif task_cfg.task_type == "binary":
            loss = "bce"
        else:
            loss = "cross_entropy"

        definition = TaskDefinition(
            name=task_name,
            task_type=task_cfg.task_type,
            num_labels=task_cfg.num_labels,
            loss=loss,
            loss_weight=1.0,
            threshold=0.5,
            auto_threshold=(task_cfg.task_type == "multilabel"),
        )

        _validate_task(definition)

        _TASK_REGISTRY[task_name] = definition

    if not _TASK_REGISTRY:
        raise RuntimeError("Task registry is empty")

    logger.info("Task registry initialized | %d tasks", len(_TASK_REGISTRY))


# Build on import (safe due to caching)
_build_registry()


# =========================================================
# PUBLIC API (USED ACROSS SYSTEM)
# =========================================================

def get_task(task: str) -> TaskDefinition:
    return _TASK_REGISTRY[task]


def get_all_tasks():
    return list(_TASK_REGISTRY.keys())


def get_task_type(task: str) -> str:
    return _TASK_REGISTRY[task].task_type


def get_output_dim(task: str) -> int:
    return _TASK_REGISTRY[task].num_labels


def get_loss_name(task: str) -> str:
    return _TASK_REGISTRY[task].loss


def get_loss_weight(task: str) -> float:
    return _TASK_REGISTRY[task].loss_weight


def get_threshold(task: str) -> float:
    return _TASK_REGISTRY[task].threshold


def use_auto_threshold(task: str) -> bool:
    return _TASK_REGISTRY[task].auto_threshold


def is_multilabel(task: str) -> bool:
    return _TASK_REGISTRY[task].task_type == "multilabel"


def is_binary(task: str) -> bool:
    return _TASK_REGISTRY[task].task_type == "binary"


def is_multiclass(task: str) -> bool:
    return _TASK_REGISTRY[task].task_type == "multiclass"