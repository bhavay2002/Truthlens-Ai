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


def tokenize_function(example, tokenizer):
    return tokenizer(
        example["text"],
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
# TRAINING
# -------------------------------------------------------

def train_model(df: pd.DataFrame, params: dict[str, Any] | None = None):

    set_seed(SEED)
    params = params or {}

    learning_rate = float(params.get("learning_rate", DEFAULT_LEARNING_RATE))
    batch_size = int(params.get("batch_size", DEFAULT_BATCH_SIZE))
    epochs = int(params.get("epochs", DEFAULT_EPOCHS))

    train_df, val_df, test_df = train_test_split(df, test_size=0.2, random_state=SEED)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    data_collator = DataCollatorWithPadding(
        tokenizer=tokenizer,
        pad_to_multiple_of=8
    )

    train_dataset = _to_hf_dataset(train_df)
    val_dataset = _to_hf_dataset(val_df)

    map_num_proc = min(4, os.cpu_count() or 1)

    train_dataset = train_dataset.map(
        lambda x: tokenize_function(x, tokenizer),
        batched=True,
        num_proc=map_num_proc,
    )
    val_dataset = val_dataset.map(
        lambda x: tokenize_function(x, tokenizer),
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

    training_args = TrainingArguments(

        output_dir=str(MODELS_DIR),

        learning_rate=learning_rate,

        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,

        gradient_accumulation_steps=2,

        num_train_epochs=epochs,

        logging_dir=str(LOGS_DIR),
        logging_steps=200,

        save_strategy="epoch",
        evaluation_strategy="epoch",

        load_best_model_at_end=True,

        bf16=torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
        fp16=not torch.cuda.is_bf16_supported(),

        optim="adamw_torch_fused",

        dataloader_num_workers=4,
        dataloader_pin_memory=True,
        dataloader_prefetch_factor=2,
        dataloader_persistent_workers=True,

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

    trainer.train(resume_from_checkpoint=get_last_checkpoint(MODELS_DIR))

    trainer.save_model(str(MODEL_PATH))
    tokenizer.save_pretrained(str(MODEL_PATH))

    return trainer, test_df