from __future__ import annotations

import torch
from torch.nn.utils.rnn import pad_sequence
from typing import List, Dict, Any


# =========================================================
# CORE COLLATE
# =========================================================

def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Generic collate function for all tasks.

    Expected input:
        [
            {
                "input_ids": Tensor(seq_len),
                "attention_mask": Tensor(seq_len),
                "labels": Tensor(...),
                "task": str
            },
            ...
        ]
    """

    if not batch:
        raise ValueError("Empty batch")

    # -----------------------------------------------------
    # INPUTS
    # -----------------------------------------------------

    input_ids = pad_sequence(
        [item["input_ids"] for item in batch],
        batch_first=True,
        padding_value=0,
    )

    attention_mask = pad_sequence(
        [item["attention_mask"] for item in batch],
        batch_first=True,
        padding_value=0,
    )

    # -----------------------------------------------------
    # LABELS
    # -----------------------------------------------------

    labels = batch[0]["labels"]

    if isinstance(labels, torch.Tensor):

        # classification (scalar) OR multilabel (vector)
        labels = torch.stack([item["labels"] for item in batch])

    else:
        raise TypeError("Unsupported label type")

    # -----------------------------------------------------
    # TASK (CRITICAL)
    # -----------------------------------------------------

    # all samples in batch must belong to same task
    task = batch[0]["task"]

    # safety check
    for item in batch:
        if item["task"] != task:
            raise RuntimeError("Mixed-task batch detected")

    # -----------------------------------------------------
    # OUTPUT
    # -----------------------------------------------------

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
        "task": task,
    }


# =========================================================
# OPTIONAL: FAST COLLATE (NO CHECKS)
# =========================================================

def fast_collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Faster version (no safety checks).
    Use only when confident about data correctness.
    """

    input_ids = pad_sequence(
        [item["input_ids"] for item in batch],
        batch_first=True,
        padding_value=0,
    )

    attention_mask = pad_sequence(
        [item["attention_mask"] for item in batch],
        batch_first=True,
        padding_value=0,
    )

    labels = torch.stack([item["labels"] for item in batch])

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
        "task": batch[0]["task"],
    }