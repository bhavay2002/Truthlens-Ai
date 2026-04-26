"""
Data Contracts for TruthLens

Defines strict schemas for each task.
Used by:
- data_validator
- dataset_factory
- training pipeline
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Dict, Optional


# =========================================================
# BASE CONTRACT
# =========================================================

@dataclass(frozen=True)
class DataContract:
    task: str
    task_type: str  # classification | multilabel

    text_column: str

    # labels
    label_columns: List[str]

    # classification only
    num_classes: Optional[int] = None

    # optional metadata columns
    optional_columns: Optional[List[str]] = None


# =========================================================
# TASK CONTRACTS
# =========================================================

CONTRACTS: Dict[str, DataContract] = {

    # -----------------------------------------------------
    # SINGLE-LABEL TASKS
    # -----------------------------------------------------

    "bias": DataContract(
        task="bias",
        task_type="classification",
        text_column="text",
        label_columns=["bias_label"],
        num_classes=2,
    ),

    "ideology": DataContract(
        task="ideology",
        task_type="classification",
        text_column="text",
        label_columns=["ideology_label"],
        num_classes=3,
    ),

    "propaganda": DataContract(
        task="propaganda",
        task_type="classification",
        text_column="text",
        label_columns=["propaganda_label"],
        num_classes=2,
    ),

    # -----------------------------------------------------
    # MULTI-LABEL TASKS
    # -----------------------------------------------------

    "frame": DataContract(
        task="frame",
        task_type="multilabel",
        text_column="text",
        label_columns=["CO", "EC", "HI", "MO", "RE"],
    ),

    "narrative": DataContract(
        task="narrative",
        task_type="multilabel",
        text_column="text",
        label_columns=["hero", "villain", "victim"],
        optional_columns=[
            "hero_entities",
            "villain_entities",
            "victim_entities",
        ],
    ),

    "emotion": DataContract(
        task="emotion",
        task_type="multilabel",
        text_column="text",
        label_columns=[f"emotion_{i}" for i in range(20)],
    ),
}


# =========================================================
# ACCESS HELPERS
# =========================================================

def get_contract(task: str) -> DataContract:
    if task not in CONTRACTS:
        raise ValueError(f"Unknown task: {task}")
    return CONTRACTS[task]


def list_tasks() -> List[str]:
    return list(CONTRACTS.keys())


# =========================================================
# VALIDATION HELPERS
# =========================================================

def get_required_columns(task: str) -> List[str]:
    contract = get_contract(task)
    return [contract.text_column] + contract.label_columns


def get_optional_columns(task: str) -> List[str]:
    contract = get_contract(task)
    return contract.optional_columns or []


def is_multilabel(task: str) -> bool:
    return get_contract(task).task_type == "multilabel"


def is_classification(task: str) -> bool:
    return get_contract(task).task_type == "classification"


def get_num_classes(task: str) -> Optional[int]:
    return get_contract(task).num_classes


# =========================================================
# DEBUG / INSPECTION
# =========================================================

def describe_contract(task: str) -> Dict:
    c = get_contract(task)

    return {
        "task": c.task,
        "type": c.task_type,
        "text_column": c.text_column,
        "labels": c.label_columns,
        "num_classes": c.num_classes,
        "optional": c.optional_columns,
    }