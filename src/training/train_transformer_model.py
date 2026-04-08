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
    torch.backends.cuda.enable_flash_sdp(True)
    torch.backends.cuda.enable_mem_efficient_sdp(True)
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

LOCAL_MODELS_DIR = Path("/content/models_local")
LOCAL_MODELS_DIR.mkdir(parents=True, exist_ok=True)

DRIVE_MODELS_DIR = Path("/content/drive/MyDrive/truthlens-models")
DRIVE_MODELS_DIR.mkdir(parents=True, exist_ok=True)

MODELS_DIR = LOCAL_MODELS_DIR

TOKENIZED_DATASET_CACHE_DIR = Path("/content/tokenized_dataset")

LOGS_DIR = Path("/content/drive/MyDrive/truthlens-logs")
LOGS_DIR.mkdir(parents=True, exist_ok=True)

MODEL_PATH = LOCAL_MODELS_DIR / "final_model"

# -------------------------------------------------------
# HELPERS
# -------------------------------------------------------

def _sync_models_to_drive():

    try:

        if DRIVE_MODELS_DIR.exists():
            shutil.rmtree(DRIVE_MODELS_DIR)

        shutil.copytree(LOCAL_MODELS_DIR, DRIVE_MODELS_DIR)

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


def _split_train_val_test(df, label_column="label"):

    holdout_size = DEFAULT_VALIDATION_SIZE + DEFAULT_TEST_SIZE

    train_df, holdout_df = train_test_split(
        df,
        test_size=holdout_size,
        random_state=SEED,
        stratify=df[label_column],
    )

    val_fraction = DEFAULT_VALIDATION_SIZE / holdout_size

    val_df, test_df = train_test_split(
        holdout_df,
        test_size=(1.0 - val_fraction),
        random_state=SEED,
        stratify=holdout_df[label_column],
    )

    return train_df.reset_index(drop=True), val_df.reset_index(drop=True), test_df.reset_index(drop=True)


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

    ensure_dataframe(df, name="df", required_columns=[text_column, label_column], min_rows=2)
    ensure_non_empty_text_column(df, text_column)

    train_df, val_df, test_df = _split_train_val_test(df)

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

    def tokenize(example):

        return tokenizer(
            example[text_column],
            truncation=True,
            padding=False,
            max_length=MAX_LENGTH,
        )

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

        train_dataset = train_dataset.map(tokenize, batched=True, num_proc=2)
        val_dataset = val_dataset.map(tokenize, batched=True, num_proc=2)

        train_dataset = train_dataset.map(compute_length, num_proc=2)
        val_dataset = val_dataset.map(compute_length, num_proc=2)

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

    training_args = TrainingArguments(

        output_dir=str(MODELS_DIR),

        learning_rate=learning_rate,

        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,

        num_train_epochs=epochs,

        logging_dir=str(LOGS_DIR),
        logging_steps=200,

        save_strategy="steps",
        save_steps=200,
        save_total_limit=3,

        evaluation_strategy="steps",
        eval_steps=200,

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