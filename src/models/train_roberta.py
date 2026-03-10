"""
RoBERTa Training Module for TruthLens AI
Handles dataset splitting, tokenization, training, checkpointing, and model saving
"""

import inspect
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score
from sklearn.model_selection import train_test_split
from transformers import (
    EarlyStoppingCallback,
    RobertaForSequenceClassification,
    RobertaTokenizer,
    Trainer,
    TrainingArguments,
)

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

MODELS_DIR = Path(SETTINGS.paths.models_dir)
LOGS_DIR = Path(SETTINGS.paths.logs_dir)
MODEL_PATH = Path(SETTINGS.model.path)
TEST_SET_PATH = Path(SETTINGS.data.test_set_path)

ID2LABEL = {0: "REAL", 1: "FAKE"}
LABEL2ID = {"REAL": 0, "FAKE": 1}


# -------------------------------------------------
# Evaluation Metrics
# -------------------------------------------------

def compute_metrics(eval_pred):
    logits, labels = eval_pred

    preds = np.argmax(logits, axis=1)

    precision, recall, f1, _ = precision_recall_fscore_support(
        labels,
        preds,
        average="binary",
    )

    acc = accuracy_score(labels, preds)

    try:
        probs = torch.softmax(torch.tensor(logits), dim=1)[:, 1].numpy()
        roc_auc = roc_auc_score(labels, probs)
    except Exception:
        roc_auc = 0.0

    return {
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "roc_auc": roc_auc,
    }


# -------------------------------------------------
# Tokenization
# -------------------------------------------------

def tokenize_function(example, tokenizer, text_column: str):
    return tokenizer(
        example[text_column],
        truncation=True,
        padding="max_length",
        max_length=MAX_LENGTH,
    )


# -------------------------------------------------
# Helper — Find Latest Checkpoint
# -------------------------------------------------

def get_last_checkpoint(directory: Path):
    if not directory.exists():
        return None

    checkpoints = list(directory.glob("checkpoint-*"))
    if not checkpoints:
        return None

    checkpoints = sorted(checkpoints, key=lambda x: int(x.name.split("-")[-1]))
    return str(checkpoints[-1])


# -------------------------------------------------
# Training Function
# -------------------------------------------------

def train_model(
    df: pd.DataFrame,
    params: dict[str, Any] | None = None,
    text_column: str = "text",
    validation_df: pd.DataFrame | None = None,
    test_df: pd.DataFrame | None = None,
):
    try:

        ensure_dataframe(df, name="df", required_columns=[text_column, "label"], min_rows=2)
        ensure_non_empty_text_column(df, text_column, name="df")

        logger.info("Starting model training...")

        torch.manual_seed(SEED)
        np.random.seed(SEED)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info("Using device: %s", device)

        # ------------------------------
        # Safe parameter handling
        # ------------------------------

        params = params or {}

        learning_rate = params.get("learning_rate", DEFAULT_LEARNING_RATE)
        batch_size = params.get("batch_size", DEFAULT_BATCH_SIZE)
        epochs = params.get("epochs", DEFAULT_EPOCHS)

        logger.info(
            "Training configuration -> LR: %s, Batch Size: %s, Epochs: %s",
            learning_rate,
            batch_size,
            epochs,
        )

        # ------------------------------
        # Dataset Split
        # ------------------------------

        train_df, temp_df = train_test_split(
            df,
            test_size=0.3,
            random_state=SEED,
            stratify=df["label"],
        )

        val_df, test_df = train_test_split(
            temp_df,
            test_size=0.5,
            random_state=SEED,
            stratify=temp_df["label"],
        )

        logger.info(
            "Train samples: %s, Val samples: %s, Test samples: %s",
            len(train_df),
            len(val_df),
            len(test_df),
        )

        # ------------------------------
        # Tokenizer
        # ------------------------------

        tokenizer = RobertaTokenizer.from_pretrained(MODEL_NAME)

        train_dataset = Dataset.from_pandas(train_df)
        val_dataset = Dataset.from_pandas(val_df)
        test_dataset = Dataset.from_pandas(test_df)

        # Remove extra index column if present
        for dataset in [train_dataset, val_dataset, test_dataset]:
            if "__index_level_0__" in dataset.column_names:
                dataset = dataset.remove_columns(["__index_level_0__"])

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

        train_dataset.set_format(
            type="torch",
            columns=["input_ids", "attention_mask", "label"],
        )

        val_dataset.set_format(
            type="torch",
            columns=["input_ids", "attention_mask", "label"],
        )

        test_dataset.set_format(
            type="torch",
            columns=["input_ids", "attention_mask", "label"],
        )

        # ------------------------------
        # Model
        # ------------------------------

        model = RobertaForSequenceClassification.from_pretrained(
            MODEL_NAME,
            num_labels=2,
            id2label=ID2LABEL,
            label2id=LABEL2ID,
        )

        model.to(device)

        # ------------------------------
        # Training Arguments
        # ------------------------------

        training_kwargs = {
            "output_dir": str(MODELS_DIR),
            "learning_rate": learning_rate,
            "weight_decay": 0.01,
            "per_device_train_batch_size": batch_size,
            "per_device_eval_batch_size": batch_size,
            "gradient_accumulation_steps": 2,
            "num_train_epochs": epochs,
            "save_strategy": "epoch",
            "logging_dir": str(LOGS_DIR),
            "logging_steps": 100,
            "load_best_model_at_end": True,
            "metric_for_best_model": "f1",
            "save_total_limit": 2,
            "fp16": torch.cuda.is_available(),
            "seed": SEED,
        }

        # Handle transformers version compatibility
        if "evaluation_strategy" in inspect.signature(TrainingArguments).parameters:
            training_kwargs["evaluation_strategy"] = "epoch"
        else:
            training_kwargs["eval_strategy"] = "epoch"

        training_args = TrainingArguments(**training_kwargs)

        # ------------------------------
        # Trainer
        # ------------------------------

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            compute_metrics=compute_metrics,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
        )

        # ------------------------------
        # Resume From Checkpoint
        # ------------------------------

        last_checkpoint = get_last_checkpoint(MODELS_DIR)

        if last_checkpoint:
            logger.info("Resuming training from checkpoint: %s", last_checkpoint)
            trainer.train(resume_from_checkpoint=last_checkpoint)
        else:
            logger.info("No checkpoint found. Starting fresh training.")
            trainer.train()

        # ------------------------------
        # Save Model
        # ------------------------------

        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        LOGS_DIR.mkdir(parents=True, exist_ok=True)
        MODEL_PATH.mkdir(parents=True, exist_ok=True)

        trainer.save_model(str(MODEL_PATH))
        tokenizer.save_pretrained(str(MODEL_PATH))

        test_df.to_csv(TEST_SET_PATH, index=False)

        logger.info("Training complete!")

        return trainer, test_dataset

    except Exception as e:
        logger.error("Error during training: %s", e)
        raise