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
from pathlib import Path
from typing import Any, Tuple

import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score
from sklearn.model_selection import train_test_split
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)
from transformers.trainer_utils import get_last_checkpoint as hf_get_last_checkpoint

from src.utils.input_validation import ensure_dataframe, ensure_non_empty_text_column
from src.utils.settings import load_settings

logger = logging.getLogger(__name__)

SETTINGS = load_settings()

MODEL_NAME = SETTINGS.model.name
MAX_LENGTH = SETTINGS.model.max_length

SEED = SETTINGS.training.seed
DEFAULT_EPOCHS = SETTINGS.training.epochs
DEFAULT_BATCH_SIZE = SETTINGS.training.batch_size
DEFAULT_LEARNING_RATE = SETTINGS.training.learning_rate
DEFAULT_RESUME_FROM_CHECKPOINT = SETTINGS.training.resume_from_checkpoint

DEFAULT_VALIDATION_SIZE = SETTINGS.training.validation_size
DEFAULT_TEST_SIZE = SETTINGS.training.test_size

MODELS_DIR = Path(SETTINGS.paths.models_dir)
LOGS_DIR = Path(SETTINGS.paths.logs_dir)
MODEL_PATH = Path(SETTINGS.model.path)
TEST_SET_PATH = Path(SETTINGS.data.test_set_path)


def compute_metrics(eval_pred: Tuple[np.ndarray, np.ndarray]) -> dict:
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


def tokenize_function(example: dict, tokenizer, text_column: str):
    return tokenizer(
        example[text_column],
        truncation=True,
        padding="max_length",
        max_length=MAX_LENGTH,
    )


def get_last_checkpoint(directory: Path) -> str | None:
    if not directory.exists():
        return None
    try:
        return hf_get_last_checkpoint(str(directory))
    except Exception:
        return None


def _split_train_val_test(
    df: pd.DataFrame,
    *,
    label_column: str = "label",
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:

    holdout_size = DEFAULT_VALIDATION_SIZE + DEFAULT_TEST_SIZE

    if not (0.0 < holdout_size < 1.0):
        raise ValueError("validation_size + test_size must be between 0 and 1")

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

    return (
        train_df.reset_index(drop=True),
        val_df.reset_index(drop=True),
        test_df.reset_index(drop=True),
    )


def _validate_split_df(
    df: pd.DataFrame,
    name: str,
    text_column: str,
    label_column: str = "label",
):
    ensure_dataframe(df, name=name, required_columns=[text_column, label_column], min_rows=2)
    ensure_non_empty_text_column(df, text_column, name=name)


def _to_hf_dataset(df: pd.DataFrame) -> Dataset:
    dataset = Dataset.from_pandas(df.reset_index(drop=True))

    if "__index_level_0__" in dataset.column_names:
        dataset = dataset.remove_columns(["__index_level_0__"])

    return dataset


def _compute_checkpoint_save_steps(
    *,
    train_examples: int,
    batch_size: int,
    gradient_accumulation_steps: int,
    epochs: int,
) -> int:

    forward_steps_per_epoch = math.ceil(train_examples / batch_size)
    optimizer_steps_per_epoch = math.ceil(forward_steps_per_epoch / gradient_accumulation_steps)
    total_optimizer_steps = max(1, optimizer_steps_per_epoch * epochs)

    return max(1, math.ceil(total_optimizer_steps * 0.10))


def train_model(
    df: pd.DataFrame,
    params: dict[str, Any] | None = None,
    text_column: str = "text",
    label_column: str = "label",
    validation_df: pd.DataFrame | None = None,
    test_df: pd.DataFrame | None = None,
):
    """
    Train TruthLens transformer classifier.

    Returns
    -------
    Trainer
    Dataset (test)
    """

    try:

        logger.info("Starting TruthLens transformer training pipeline")

        _validate_split_df(df, "df", text_column, label_column)

        if validation_df is None or test_df is None:
            train_df, val_df, resolved_test_df = _split_train_val_test(
                df,
                label_column=label_column,
            )
        else:
            train_df = df
            val_df = validation_df
            resolved_test_df = test_df

        torch.manual_seed(SEED)
        np.random.seed(SEED)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info("Training device: %s", device)

        params = params or {}

        learning_rate = float(params.get("learning_rate", DEFAULT_LEARNING_RATE))
        batch_size = int(params.get("batch_size", DEFAULT_BATCH_SIZE))
        epochs = int(params.get("epochs", DEFAULT_EPOCHS))
        gradient_accumulation_steps = 2

        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

        train_dataset = _to_hf_dataset(train_df)
        val_dataset = _to_hf_dataset(val_df)
        test_dataset = _to_hf_dataset(resolved_test_df)

        train_dataset = train_dataset.map(
            lambda x: tokenize_function(x, tokenizer, text_column),
            batched=True,
        )

        val_dataset = val_dataset.map(
            lambda x: tokenize_function(x, tokenizer, text_column),
            batched=True,
        )

        test_dataset = test_dataset.map(
            lambda x: tokenize_function(x, tokenizer, text_column),
            batched=True,
        )

        train_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])
        val_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])
        test_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])

        num_labels = len(df[label_column].unique())

        model = AutoModelForSequenceClassification.from_pretrained(
            MODEL_NAME,
            num_labels=num_labels,
        )

        model.to(device)

        checkpoint_save_steps = _compute_checkpoint_save_steps(
            train_examples=len(train_df),
            batch_size=batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            epochs=epochs,
        )

        training_args = TrainingArguments(
            output_dir=str(MODELS_DIR),
            learning_rate=learning_rate,
            weight_decay=0.01,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            num_train_epochs=epochs,
            logging_dir=str(LOGS_DIR),
            logging_steps=max(1, min(100, checkpoint_save_steps)),
            save_strategy="steps",
            save_steps=checkpoint_save_steps,
            evaluation_strategy="steps",
            eval_steps=checkpoint_save_steps,
            load_best_model_at_end=True,
            metric_for_best_model="f1",
            save_total_limit=3,
            fp16=torch.cuda.is_available(),
            seed=SEED,
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            compute_metrics=compute_metrics,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
        )

        last_checkpoint = None
        if DEFAULT_RESUME_FROM_CHECKPOINT:
            last_checkpoint = get_last_checkpoint(MODELS_DIR)

        trainer.train(resume_from_checkpoint=last_checkpoint)

        trainer.save_model(str(MODEL_PATH))
        tokenizer.save_pretrained(str(MODEL_PATH))

        resolved_test_df.to_csv(TEST_SET_PATH, index=False)

        logger.info("Training completed successfully")

        return trainer, test_dataset

    except Exception:
        logger.exception("Training pipeline failed")
        raise