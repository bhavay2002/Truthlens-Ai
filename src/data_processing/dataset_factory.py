from __future__ import annotations

import logging
from typing import Any

import pandas as pd

from src.data.data_contracts import get_contract
from src.data.dataset import ClassificationDataset, MultiLabelDataset

logger = logging.getLogger(__name__)


# =========================================================
# FACTORY
# =========================================================

def build_dataset(
    *,
    task: str,
    df: pd.DataFrame,
    tokenizer: Any,
    max_length: int = 512,
):
    """
    Build dataset based on task contract.

    Args:
        task: task name (bias, ideology, etc.)
        df: input dataframe
        tokenizer: HF tokenizer
        max_length: token max length

    Returns:
        Dataset instance
    """

    contract = get_contract(task)

    logger.info(
        "Building dataset | task=%s | type=%s | rows=%d",
        task,
        contract.task_type,
        len(df),
    )

    # -----------------------------------------------------
    # CLASSIFICATION
    # -----------------------------------------------------

    if contract.task_type == "classification":

        label_col = contract.label_columns[0]

        return ClassificationDataset(
            df=df,
            tokenizer=tokenizer,
            label_col=label_col,
            num_classes=contract.num_classes,
            max_length=max_length,
        )

    # -----------------------------------------------------
    # MULTILABEL
    # -----------------------------------------------------

    elif contract.task_type == "multilabel":

        return MultiLabelDataset(
            df=df,
            tokenizer=tokenizer,
            label_cols=contract.label_columns,
            task_name=task,
            max_length=max_length,
        )

    else:
        raise ValueError(f"Unsupported task type: {contract.task_type}")


# =========================================================
# BULK FACTORY (MULTI-TASK)
# =========================================================

def build_all_datasets(
    *,
    datasets: dict,
    tokenizer: Any,
    max_length: int = 512,
):
    """
    Build datasets for all tasks.

    Args:
        datasets:
            {
                "bias": {"train": df, "val": df, "test": df},
                ...
            }

    Returns:
        same structure but with Dataset objects
    """

    result = {}

    for task, splits in datasets.items():

        result[task] = {}

        for split, df in splits.items():

            result[task][split] = build_dataset(
                task=task,
                df=df,
                tokenizer=tokenizer,
                max_length=max_length,
            )

    return result


# =========================================================
# VALIDATION (OPTIONAL SAFETY)
# =========================================================

def validate_dataset_compatibility(task: str, df: pd.DataFrame):
    """
    Ensure dataframe matches contract before dataset creation.
    """

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