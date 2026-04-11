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
import shutil
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
# GPU PERFORMANCE SETTINGS
# -------------------------------------------------------

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.set_float32_matmul_precision("high")

if torch.cuda.is_available():
    if hasattr(torch.backends.cuda, "enable_flash_sdp"):
        torch.backends.cuda.enable_flash_sdp(True)
    if hasattr(torch.backends.cuda, "enable_mem_efficient_sdp"):
        torch.backends.cuda.enable_mem_efficient_sdp(True)
    if hasattr(torch.backends.cuda, "enable_math_sdp"):
        torch.backends.cuda.enable_math_sdp(True)

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
GOOGLE_DRIVE_REPORTS_DIR = SETTINGS.paths.reports_dir
GOOGLE_DRIVE_CHECKPOINTS_DIR = MODELS_DIR / "checkpoints"
MODEL_PATH = SETTINGS.model.path
TOKENIZED_DATASET_CACHE_DIR = MODELS_DIR / "tokenized_dataset"

MODELS_DIR.mkdir(parents=True, exist_ok=True)
LOGS_DIR.mkdir(parents=True, exist_ok=True)
GOOGLE_DRIVE_REPORTS_DIR.mkdir(parents=True, exist_ok=True)
GOOGLE_DRIVE_CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)
MODEL_PATH.mkdir(parents=True, exist_ok=True)

# -------------------------------------------------------
# HELPERS
# -------------------------------------------------------

def _sync_models_to_drive():

    try:

        if GOOGLE_DRIVE_CHECKPOINTS_DIR.exists() and GOOGLE_DRIVE_CHECKPOINTS_DIR != MODELS_DIR:
            shutil.rmtree(GOOGLE_DRIVE_CHECKPOINTS_DIR)

        if GOOGLE_DRIVE_CHECKPOINTS_DIR != MODELS_DIR:
            shutil.copytree(MODELS_DIR, GOOGLE_DRIVE_CHECKPOINTS_DIR)

        logger.info("Models synced to Google Drive")

    except Exception as exc:

        logger.warning("Drive sync failed: %s", exc)


def compute_metrics(eval_pred: Tuple[np.ndarray, np.ndarray]):

    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)

    unique_labels = np.unique(labels)
    average = "binary" if len(unique_labels) <= 2 else "macro"

    precision, recall, f1, _ = precision_recall_fscore_support(
        labels,
        preds,
        average=average,
        zero_division=0,
    )

    acc = accuracy_score(labels, preds)

    try:

        probs = torch.softmax(torch.tensor(logits), dim=1).numpy()

        if probs.shape[1] == 2:
            roc_auc = roc_auc_score(labels, probs[:, 1])
        else:
            roc_auc = roc_auc_score(labels, probs, multi_class="ovr")

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


def _validate_split_df(
    df: pd.DataFrame,
    df_name: str,
    text_column: str,
    label_column: str = "label",
) -> None:

    ensure_dataframe(
        df,
        name=df_name,
        required_columns=[text_column, label_column],
        min_rows=2,
    )
    ensure_non_empty_text_column(df, text_column)


def _should_stratify(df: pd.DataFrame, label_column: str) -> bool:

    label_counts = df[label_column].value_counts(dropna=False)

    return len(label_counts) > 1 and bool((label_counts >= 2).all())


def _split_train_val_test(df: pd.DataFrame, label_column: str = "label"):

    _validate_split_df(df, "df", text_column="text", label_column=label_column)

    holdout_size = DEFAULT_VALIDATION_SIZE + DEFAULT_TEST_SIZE
    if not 0 < holdout_size < 1:
        raise ValueError("validation_size + test_size must be in (0, 1)")

    stratify_values = df[label_column] if _should_stratify(df, label_column) else None

    train_df, holdout_df = train_test_split(
        df,
        test_size=holdout_size,
        random_state=SEED,
        stratify=stratify_values,
    )

    val_fraction = DEFAULT_VALIDATION_SIZE / holdout_size

    holdout_stratify = (
        holdout_df[label_column]
        if _should_stratify(holdout_df, label_column)
        else None
    )

    val_df, test_df = train_test_split(
        holdout_df,
        test_size=(1.0 - val_fraction),
        random_state=SEED,
        stratify=holdout_stratify,
    )

    return train_df.reset_index(drop=True), val_df.reset_index(drop=True), test_df.reset_index(drop=True)


def tokenize_function(
    example: dict[str, Any],
    tokenizer: AutoTokenizer,
    text_column: str = "text",
    max_length: int = MAX_LENGTH,
) -> dict[str, Any]:

    return tokenizer(
        example[text_column],
        truncation=True,
        padding=False,
        max_length=max_length,
    )


def _compute_checkpoint_save_steps(
    train_examples: int,
    batch_size: int,
    gradient_accumulation_steps: int,
    epochs: int,
) -> int:

    if train_examples <= 0:
        raise ValueError("train_examples must be > 0")
    if batch_size <= 0:
        raise ValueError("batch_size must be > 0")
    if gradient_accumulation_steps <= 0:
        raise ValueError("gradient_accumulation_steps must be > 0")
    if epochs <= 0:
        raise ValueError("epochs must be > 0")

    forward_steps_per_epoch = math.ceil(train_examples / batch_size)
    optimizer_steps_per_epoch = math.ceil(
        forward_steps_per_epoch / gradient_accumulation_steps
    )
    total_optimizer_steps = optimizer_steps_per_epoch * epochs

    return max(1, math.ceil(total_optimizer_steps * 0.1))


def get_last_checkpoint(directory: Path):

    if not directory.exists():
        return None

    try:
        return hf_get_last_checkpoint(str(directory))
    except Exception:
        return None


# -------------------------------------------------------
# TRAINING
# -------------------------------------------------------

def train_model(
    df: pd.DataFrame,
    params: dict[str, Any] | None = None,
    text_column="text",
    label_column="label",
):

    logger.info("Starting TruthLens training")

    _validate_split_df(df, "df", text_column=text_column, label_column=label_column)

    train_df, val_df, test_df = _split_train_val_test(df, label_column=label_column)

    set_seed(SEED)

    params = params or {}

    learning_rate = float(params.get("learning_rate", DEFAULT_LEARNING_RATE))
    batch_size = int(params.get("batch_size", DEFAULT_BATCH_SIZE))
    epochs = int(params.get("epochs", DEFAULT_EPOCHS))

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    train_dataset = _to_hf_dataset(train_df)
    val_dataset = _to_hf_dataset(val_df)
    test_dataset = _to_hf_dataset(test_df)

    def compute_length(example):
        return {"length": len(example["input_ids"])}

    cache_train_dir = TOKENIZED_DATASET_CACHE_DIR / "train"
    cache_val_dir = TOKENIZED_DATASET_CACHE_DIR / "val"

    if (
        cache_train_dir.exists()
        and cache_val_dir.exists()
        and (cache_train_dir / "dataset_info.json").exists()
    ):

        train_dataset = load_from_disk(str(cache_train_dir))
        val_dataset = load_from_disk(str(cache_val_dir))

    else:

        map_num_proc = 1

        train_dataset = train_dataset.map(
            lambda ex: tokenize_function(
                ex,
                tokenizer=tokenizer,
                text_column=text_column,
                max_length=MAX_LENGTH,
            ),
            batched=True,
            num_proc=map_num_proc,
        )
        val_dataset = val_dataset.map(
            lambda ex: tokenize_function(
                ex,
                tokenizer=tokenizer,
                text_column=text_column,
                max_length=MAX_LENGTH,
            ),
            batched=True,
            num_proc=map_num_proc,
        )

        train_dataset = train_dataset.map(compute_length, num_proc=map_num_proc)
        val_dataset = val_dataset.map(compute_length, num_proc=map_num_proc)

        cache_train_dir.parent.mkdir(parents=True, exist_ok=True)

        train_dataset.save_to_disk(str(cache_train_dir))
        val_dataset.save_to_disk(str(cache_val_dir))

    model = TruthLensMultiTaskModel(MODEL_NAME)

    device = get_device()

    model.to(device)

    if torch.cuda.is_available() and torch.__version__.startswith("2"):

        try:
            model = torch.compile(model, mode="reduce-overhead")
        except Exception as exc:
            logger.warning("torch.compile skipped: %s", exc)

    save_steps = _compute_checkpoint_save_steps(
        train_examples=len(train_df),
        batch_size=batch_size,
        gradient_accumulation_steps=1,
        epochs=epochs,
    )

    training_args = TrainingArguments(

        output_dir=str(MODELS_DIR),

        learning_rate=learning_rate,

        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,

        num_train_epochs=epochs,

        logging_dir=str(LOGS_DIR),
        logging_steps=200,

        save_strategy="steps",
        save_steps=save_steps,
        save_total_limit=3,

        evaluation_strategy="steps",
        eval_steps=save_steps,

        load_best_model_at_end=True,

        fp16=torch.cuda.is_available(),

        dataloader_num_workers=2,
        dataloader_pin_memory=True,

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

    last_checkpoint = get_last_checkpoint(MODELS_DIR)

    trainer.train(resume_from_checkpoint=last_checkpoint)

    trainer.save_model(str(MODEL_PATH))

    tokenizer.save_pretrained(str(MODEL_PATH))

    _sync_models_to_drive()

    logger.info("Training completed")

    return trainer, test_dataset