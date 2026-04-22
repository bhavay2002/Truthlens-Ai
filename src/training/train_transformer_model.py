"""
File Name: train_truthlens_model.py
Module: training
Description:
    Production training pipeline for the TruthLens transformer-based
    misinformation detection system.

    This module provides a unified training interface for transformer
    architectures including RoBERTa, DeBERTa, and Longformer. It handles
    dataset validation, train/validation/test splitting, tokenizer
    preparation, model initialization, training orchestration using the
    HuggingFace Trainer API, evaluation metrics, checkpoint management,
    and artifact persistence.

    The implementation follows research-grade ML engineering standards
    and is designed to support future model upgrades without changing
    the training pipeline.

Dependencies:
    logging
    math
    pathlib
    typing
    numpy
    pandas
    torch
    datasets
    sklearn
    transformers
    src.utils.input_validation
    src.utils.settings
Inputs:
    df : pandas.DataFrame
        Must contain:
            text : str
            label : int or categorical

    params : dict
        Optional hyperparameters

Outputs:
    transformers.Trainer
    datasets.Dataset (test)
"""
from __future__ import annotations

import logging
import math
import os
from pathlib import Path
from typing import Any, Tuple

import numpy as np
import pandas as pd
import torch
from datasets import Dataset, load_from_disk
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score
from sklearn.model_selection import train_test_split
from transformers import (
    AutoTokenizer,
    DataCollatorWithPadding,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)
from transformers.trainer_utils import get_last_checkpoint as hf_get_last_checkpoint

from src.models.multitask.multitask_model import TruthLensMultiTaskModel
from src.utils.input_validation import ensure_dataframe, ensure_non_empty_text_column
from src.utils.seed_utils import set_seed
from src.utils.settings import load_settings
from src.models.training.training_utils import get_device

logger = logging.getLogger(__name__)

# -------------------------------------------------------
# GPU PERFORMANCE SETTINGS  (M5: opt-in, called only from training entrypoint)
# -------------------------------------------------------

def configure_training_precision() -> None:
    """Enable TF32 + flash/mem-efficient SDP. Call from training entrypoint only."""
    if not torch.cuda.is_available():
        return
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")
    if hasattr(torch.backends.cuda, "enable_flash_sdp"):
        torch.backends.cuda.enable_flash_sdp(True)
    if hasattr(torch.backends.cuda, "enable_mem_efficient_sdp"):
        torch.backends.cuda.enable_mem_efficient_sdp(True)

# -------------------------------------------------------
# SETTINGS
# -------------------------------------------------------

SETTINGS = load_settings()

MODEL_NAME = SETTINGS.model.name
MAX_LENGTH = SETTINGS.model.max_length

SEED = SETTINGS.training.seed
DEFAULT_EPOCHS = SETTINGS.training.epochs
DEFAULT_BATCH_SIZE = SETTINGS.training.batch_size
DEFAULT_LEARNING_RATE = SETTINGS.training.learning_rate

DEFAULT_VALIDATION_SIZE = SETTINGS.training.validation_size
DEFAULT_TEST_SIZE = SETTINGS.training.test_size

# -------------------------------------------------------
# PATHS
# -------------------------------------------------------

MODELS_DIR = SETTINGS.paths.models_dir
LOGS_DIR = SETTINGS.paths.logs_dir
MODEL_PATH = SETTINGS.model.path
TOKENIZED_DATASET_CACHE_DIR = MODELS_DIR / "tokenized_dataset"

GOOGLE_DRIVE_REPORTS_DIR = SETTINGS.paths.reports_dir
GOOGLE_DRIVE_CHECKPOINTS_DIR = MODELS_DIR / "checkpoints"

# M7: HF Trainer checkpoint discovery requires its own dedicated subdir.
HF_OUTPUT_DIR = MODELS_DIR / "hf_trainer"
HF_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MODELS_DIR.mkdir(parents=True, exist_ok=True)
LOGS_DIR.mkdir(parents=True, exist_ok=True)
MODEL_PATH.mkdir(parents=True, exist_ok=True)

# -------------------------------------------------------
# HELPERS
# -------------------------------------------------------

def compute_metrics(eval_pred: Tuple[np.ndarray, np.ndarray]):

    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)

    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average="binary", zero_division=0
    )

    acc = accuracy_score(labels, preds)

    try:
        probs = torch.softmax(torch.tensor(logits), dim=1).numpy()
        roc_auc = roc_auc_score(labels, probs[:, 1])
    except Exception:
        roc_auc = 0.0

    return {
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "roc_auc": roc_auc,
    }


def _to_hf_dataset(df: pd.DataFrame) -> Dataset:
    dataset = Dataset.from_pandas(df.reset_index(drop=True))
    if "__index_level_0__" in dataset.column_names:
        dataset = dataset.remove_columns(["__index_level_0__"])
    return dataset


def tokenize_function(example, tokenizer, text_column: str = "text"):
    return tokenizer(
        example[text_column],
        truncation=True,
        padding=False,
        max_length=MAX_LENGTH,
    )


def get_last_checkpoint(directory: Path):
    if not directory.exists():
        return None
    try:
        return hf_get_last_checkpoint(str(directory))
    except Exception:
        return None


# -------------------------------------------------------
# HELPER UTILITIES
# -------------------------------------------------------

def _compute_checkpoint_save_steps(
    train_examples: int,
    batch_size: int,
    gradient_accumulation_steps: int,
    epochs: int,
) -> int:
    """Return the number of optimizer steps between each checkpoint save.

    Uses a 10% cadence: saves every ~10% of total training progress.
    Always returns at least 1.
    """
    forward_steps_per_epoch = math.ceil(train_examples / batch_size)
    optimizer_steps_per_epoch = math.ceil(forward_steps_per_epoch / gradient_accumulation_steps)
    total_steps = optimizer_steps_per_epoch * epochs
    save_steps = math.ceil(total_steps * 0.1)
    return max(1, save_steps)


def _validate_split_df(
    df: pd.DataFrame,
    name: str,
    text_column: str,
    label_column: str = "label",
) -> None:
    """Validate that *df* contains the required columns and non-empty text.

    Raises:
        ValueError: If a required column is missing or every text entry is blank.
    """
    if text_column not in df.columns:
        raise ValueError(f"{name}: missing required column '{text_column}'")
    if label_column not in df.columns:
        raise ValueError(f"{name}: missing required column '{label_column}'")
    if df[text_column].astype(str).str.strip().eq("").all():
        raise ValueError(f"{name}: '{text_column}' column contains only empty strings")


def _split_train_val_test(
    df: pd.DataFrame,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    label_column: str = "label",
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split *df* into train / validation / test partitions.

    Returns:
        (train_df, val_df, test_df)
    """
    text_col = "text"
    if text_col in df.columns:
        original_len = len(df)
        df = df.drop_duplicates(subset=[text_col]).reset_index(drop=True)
        if len(df) < original_len:
            logger.warning(
                "Removed %d duplicate rows before splitting",
                original_len - len(df),
            )

    test_ratio = 1.0 - train_ratio - val_ratio
    labels = df[label_column] if label_column in df.columns else None
    stratify_labels = None
    if labels is not None and labels.nunique(dropna=True) > 1:
        stratify_labels = labels
    train_df, temp_df = train_test_split(
        df,
        test_size=(val_ratio + test_ratio),
        random_state=random_state,
        stratify=stratify_labels,
    )
    relative_val = val_ratio / (val_ratio + test_ratio)
    temp_labels = temp_df[label_column] if label_column in temp_df.columns else None
    stratify_temp = None
    if temp_labels is not None and temp_labels.nunique(dropna=True) > 1:
        stratify_temp = temp_labels
    val_df, test_df = train_test_split(
        temp_df,
        test_size=(1.0 - relative_val),
        random_state=random_state,
        stratify=stratify_temp,
    )
    return train_df.reset_index(drop=True), val_df.reset_index(drop=True), test_df.reset_index(drop=True)



# -------------------------------------------------------
# TRAINING
# -------------------------------------------------------

def train_model(df: pd.DataFrame, params: dict[str, Any] | None = None):

    set_seed(SEED)
    params = params or {}

    learning_rate = float(params.get("learning_rate", DEFAULT_LEARNING_RATE))
    batch_size = int(params.get("batch_size", DEFAULT_BATCH_SIZE))
    epochs = int(params.get("epochs", DEFAULT_EPOCHS))

    train_df, val_df, test_df = _split_train_val_test(
        df,
        train_ratio=1 - (DEFAULT_VALIDATION_SIZE + DEFAULT_TEST_SIZE),
        val_ratio=DEFAULT_VALIDATION_SIZE,
        label_column="label",
        random_state=SEED,
    )

    text_col = SETTINGS.training.text_column

    _validate_split_df(train_df, "train", text_col)
    _validate_split_df(val_df, "validation", text_col)
    _validate_split_df(test_df, "test", text_col)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    data_collator = DataCollatorWithPadding(
        tokenizer=tokenizer,
        pad_to_multiple_of=8
    )

    train_dataset = _to_hf_dataset(train_df)
    val_dataset = _to_hf_dataset(val_df)

    map_num_proc = min(4, os.cpu_count() or 1)

    train_dataset = train_dataset.map(
        lambda x: tokenize_function(x, tokenizer, text_col),
        batched=True,
        num_proc=map_num_proc,
    )
    val_dataset = val_dataset.map(
        lambda x: tokenize_function(x, tokenizer, text_col),
        batched=True,
        num_proc=map_num_proc,
    )

    train_dataset = train_dataset.map(lambda x: {"length": len(x["input_ids"])})
    val_dataset = val_dataset.map(lambda x: {"length": len(x["input_ids"])})

    model = TruthLensMultiTaskModel(MODEL_NAME)

    model.gradient_checkpointing_enable()

    device = get_device()
    model.to(device)

    if torch.cuda.is_available():
        try:
            model = torch.compile(model, mode="max-autotune")
        except Exception as e:
            logger.warning(f"compile failed: {e}")

    grad_accum_steps = int(
        params.get(
            "gradient_accumulation_steps",
            SETTINGS.training.gradient_accumulation_steps,
        )
    )

    save_steps = _compute_checkpoint_save_steps(
        train_examples=len(train_df),
        batch_size=batch_size,
        gradient_accumulation_steps=grad_accum_steps,
        epochs=epochs,
    )

    use_cuda = torch.cuda.is_available()
    use_bf16 = use_cuda and torch.cuda.is_bf16_supported()
    use_fp16 = use_cuda and (not use_bf16)

    num_workers = min(4, os.cpu_count() or 1)
    use_workers = num_workers > 0

    training_args = TrainingArguments(

        output_dir=str(HF_OUTPUT_DIR),

        learning_rate=learning_rate,

        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,

        gradient_accumulation_steps=grad_accum_steps,

        num_train_epochs=epochs,

        logging_dir=str(LOGS_DIR),
        logging_steps=200,

        save_strategy="steps",
        save_steps=save_steps,
        evaluation_strategy="epoch",

        load_best_model_at_end=True,

        bf16=use_bf16,
        fp16=use_fp16,

        optim="adamw_torch_fused" if use_cuda else "adamw_torch",

        dataloader_num_workers=num_workers,
        dataloader_pin_memory=use_cuda,
        dataloader_prefetch_factor=2 if use_workers else None,
        dataloader_persistent_workers=use_workers,

        group_by_length=True,
        length_column_name="length",

        seed=SEED,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
    )

    trainer.train(resume_from_checkpoint=get_last_checkpoint(HF_OUTPUT_DIR))

    trainer.save_model(str(MODEL_PATH))
    tokenizer.save_pretrained(str(MODEL_PATH))

    # Return eval-ready HF dataset for CV/HPO contract compatibility
    return trainer, val_dataset