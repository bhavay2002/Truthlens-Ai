from __future__ import annotations

import torch
from torch.utils.data import Dataset
from typing import List, Dict, Any


# =========================================================
# BASE DATASET
# =========================================================

class BaseTextDataset(Dataset):
    """
    Base dataset for all tasks.
    Handles tokenization + common structure.
    """

    def __init__(
        self,
        df,
        tokenizer,
        *,
        text_col: str = "text",
        max_length: int = 512,
    ):
        self.df = df.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.text_col = text_col
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.df)

    def _encode(self, text: str) -> Dict[str, torch.Tensor]:

        enc = self.tokenizer(
            text,
            truncation=True,
            padding=False,
            max_length=self.max_length,
            return_tensors="pt",
        )

        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
        }


# =========================================================
# CLASSIFICATION DATASET (bias, ideology, propaganda)
# =========================================================

class ClassificationDataset(BaseTextDataset):

    def __init__(
        self,
        df,
        tokenizer,
        *,
        label_col: str,
        num_classes: int,
        **kwargs,
    ):
        super().__init__(df, tokenizer, **kwargs)

        self.label_col = label_col
        self.num_classes = num_classes

    def __getitem__(self, idx: int) -> Dict[str, Any]:

        row = self.df.iloc[idx]

        x = self._encode(row[self.text_col])

        label = int(row[self.label_col])

        return {
            **x,
            "labels": torch.tensor(label, dtype=torch.long),
            "task": self.label_col,  # important for routing
        }


# =========================================================
# MULTILABEL DATASET (frame, narrative, emotion)
# =========================================================

class MultiLabelDataset(BaseTextDataset):

    def __init__(
        self,
        df,
        tokenizer,
        *,
        label_cols: List[str],
        task_name: str,
        **kwargs,
    ):
        super().__init__(df, tokenizer, **kwargs)

        self.label_cols = label_cols
        self.task_name = task_name

    def __getitem__(self, idx: int) -> Dict[str, Any]:

        row = self.df.iloc[idx]

        x = self._encode(row[self.text_col])

        labels = torch.tensor(
            [float(row[c]) for c in self.label_cols],
            dtype=torch.float,
        )

        return {
            **x,
            "labels": labels,
            "task": self.task_name,
        }


# =========================================================
# DATASET FACTORY (CRITICAL)
# =========================================================

def build_dataset(
    *,
    task: str,
    df,
    tokenizer,
    max_length: int = 512,
):
    """
    Factory for all 6 tasks.
    """

    if task == "bias":
        return ClassificationDataset(
            df,
            tokenizer,
            label_col="bias",
            num_classes=2,
            max_length=max_length,
        )

    elif task == "ideology":
        return ClassificationDataset(
            df,
            tokenizer,
            label_col="ideology",
            num_classes=3,
            max_length=max_length,
        )

    elif task == "propaganda":
        return ClassificationDataset(
            df,
            tokenizer,
            label_col="propaganda",
            num_classes=2,
            max_length=max_length,
        )

    elif task == "frame":
        return MultiLabelDataset(
            df,
            tokenizer,
            label_cols=["CO", "EC", "HI", "MO", "RE"],
            task_name="frame",
            max_length=max_length,
        )

    elif task == "narrative":
        return MultiLabelDataset(
            df,
            tokenizer,
            label_cols=["hero", "villain", "victim"],
            task_name="narrative",
            max_length=max_length,
        )

    elif task == "emotion":
        return MultiLabelDataset(
            df,
            tokenizer,
            label_cols=[f"emotion_{i}" for i in range(20)],
            task_name="emotion",
            max_length=max_length,
        )

    else:
        raise ValueError(f"Unknown task: {task}")