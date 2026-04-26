"""
Contract-driven dataset factory.

Single entry point: ``build_dataset(task=…, df=…, tokenizer=…, …)``.
All label-column names are pulled from ``data_contracts.CONTRACTS``,
guaranteeing schema consistency across cleaning, validation, sampling
and dataset construction.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict

import pandas as pd

from src.data_processing.data_contracts import get_contract
from src.data_processing.dataset import (
    ClassificationDataset,
    MultiLabelDataset,
)

logger = logging.getLogger(__name__)


# =========================================================
# CONFIG
# =========================================================

@dataclass(frozen=True)
class DatasetBuildConfig:
    """Tunables that affect tokenization / cache key."""

    max_length: int = 512
    return_offsets_mapping: bool = False
    log_truncation: bool = True


# =========================================================
# FACTORY
# =========================================================

def build_dataset(
    *,
    task: str,
    df: pd.DataFrame,
    tokenizer: Any,
    max_length: int = 512,
    return_offsets_mapping: bool = False,
    log_truncation: bool = True,
):
    """
    Build a dataset for ``task`` from ``df`` using the canonical task contract.
    """
    contract = get_contract(task)

    logger.info(
        "Building dataset | task=%s | type=%s | rows=%d | max_length=%d",
        task,
        contract.task_type,
        len(df),
        max_length,
    )

    common_kwargs = dict(
        text_col=contract.text_column,
        max_length=max_length,
        return_offsets_mapping=return_offsets_mapping,
        log_truncation=log_truncation,
    )

    if contract.task_type == "classification":
        return ClassificationDataset(
            df=df,
            tokenizer=tokenizer,
            label_col=contract.label_columns[0],
            num_classes=contract.num_classes,
            task_name=task,
            **common_kwargs,
        )

    if contract.task_type == "multilabel":
        return MultiLabelDataset(
            df=df,
            tokenizer=tokenizer,
            label_cols=contract.label_columns,
            task_name=task,
            **common_kwargs,
        )

    raise ValueError(f"Unsupported task type: {contract.task_type}")


# =========================================================
# BULK FACTORY (MULTI-TASK)
# =========================================================

def build_all_datasets(
    *,
    datasets: Dict[str, Dict[str, pd.DataFrame]],
    tokenizer: Any,
    max_length: int = 512,
    return_offsets_mapping: bool = False,
    log_truncation: bool = True,
) -> Dict[str, Dict[str, Any]]:
    """
    Build datasets for every task / split.

    ``datasets`` shape:
        {"bias": {"train": df, "val": df, "test": df}, ...}
    """
    result: Dict[str, Dict[str, Any]] = {}

    for task, splits in datasets.items():
        result[task] = {}
        for split, df in splits.items():
            result[task][split] = build_dataset(
                task=task,
                df=df,
                tokenizer=tokenizer,
                max_length=max_length,
                return_offsets_mapping=return_offsets_mapping,
                log_truncation=log_truncation,
            )

    return result


# =========================================================
# COMPATIBILITY CHECK
# =========================================================

def validate_dataset_compatibility(task: str, df: pd.DataFrame) -> None:
    """Raise if ``df`` is missing any column required by the task contract."""
    contract = get_contract(task)

    missing = []
    if contract.text_column not in df.columns:
        missing.append(contract.text_column)
    for col in contract.label_columns:
        if col not in df.columns:
            missing.append(col)

    if missing:
        raise ValueError(
            f"Dataset mismatch for task={task}. Missing columns: {missing}"
        )
