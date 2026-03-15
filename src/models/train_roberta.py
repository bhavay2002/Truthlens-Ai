"""
File: train_roberta.py

Purpose
-------
Training pipeline for the TruthLens AI RoBERTa fake news classifier.

Responsibilities
----------------
1. Validate input datasets
2. Split dataset into train/validation/test
3. Tokenize text using RoBERTa tokenizer
4. Train a transformer-based classifier
5. Evaluate performance metrics
6. Save trained model and tokenizer
7. Manage checkpoints and resume training

Input
-----
df : pandas.DataFrame
    Must contain:
        text : str
        label : int (0=REAL, 1=FAKE)

params : dict
    Optional hyperparameters

validation_df : pandas.DataFrame | None
test_df : pandas.DataFrame | None

Output
------
trainer : transformers.Trainer
test_dataset : datasets.Dataset
"""

import logging
from pathlib import Path
from typing import Any, Tuple

import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from transformers import (
    EarlyStoppingCallback,
    RobertaForSequenceClassification,
    RobertaTokenizer,
    Trainer,
    TrainingArguments,
)
from transformers.trainer_utils import (
    get_last_checkpoint as hf_get_last_checkpoint,
)

from src.utils.input_validation import (
    ensure_dataframe,
    ensure_non_empty_text_column,
)
from src.utils.settings import load_settings

# ---------------------------------------------------------
# Logging Configuration
# ---------------------------------------------------------

logger = logging.getLogger(__name__)


# ---------------------------------------------------------
# Load Global Settings
# ---------------------------------------------------------

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


# ---------------------------------------------------------
# Label Mapping
# ---------------------------------------------------------

ID2LABEL = {0: "REAL", 1: "FAKE"}
LABEL2ID = {"REAL": 0, "FAKE": 1}


# ---------------------------------------------------------
# Evaluation Metrics
# ---------------------------------------------------------


def compute_metrics(eval_pred: Tuple[np.ndarray, np.ndarray]) -> dict:
    """
    Compute classification metrics for evaluation.

    Returns
    -------
    dict
        accuracy, precision, recall, f1, roc_auc
    """

    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)

    precision, recall, f1, _ = precision_recall_fscore_support(
        labels,
        preds,
        average="binary",
        zero_division=0,
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


# ---------------------------------------------------------
# Tokenization
# ---------------------------------------------------------


def tokenize_function(example: dict, tokenizer, text_column: str):
    """
    Tokenize input text for RoBERTa.
    """

    return tokenizer(
        example[text_column],
        truncation=True,
        padding="max_length",
        max_length=MAX_LENGTH,
    )


# ---------------------------------------------------------
# Checkpoint Detection
# ---------------------------------------------------------


def get_last_checkpoint(directory: Path) -> str | None:
    """
    Detect last HuggingFace checkpoint if available.
    """

    if not directory.exists():
        return None

    try:
        return hf_get_last_checkpoint(str(directory))
    except Exception:
        return None


# ---------------------------------------------------------
# Dataset Splitting
# ---------------------------------------------------------


def _split_train_val_test(
    df: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split dataset into train, validation, and test sets.
    """

    holdout_size = DEFAULT_VALIDATION_SIZE + DEFAULT_TEST_SIZE

    if not (0.0 < holdout_size < 1.0):
        raise ValueError("validation_size + test_size must be between 0 and 1")

    train_df, holdout_df = train_test_split(
        df,
        test_size=holdout_size,
        random_state=SEED,
        stratify=df["label"],
    )

    val_fraction = DEFAULT_VALIDATION_SIZE / holdout_size

    val_df, test_df = train_test_split(
        holdout_df,
        test_size=(1.0 - val_fraction),
        random_state=SEED,
        stratify=holdout_df["label"],
    )

    return (
        train_df.reset_index(drop=True),
        val_df.reset_index(drop=True),
        test_df.reset_index(drop=True),
    )


# ---------------------------------------------------------
# Dataset Validation
# ---------------------------------------------------------


def _validate_split_df(df: pd.DataFrame, name: str, text_column: str):
    """
    Validate dataframe integrity.
    """

    ensure_dataframe(
        df, name=name, required_columns=[text_column, "label"], min_rows=2
    )
    ensure_non_empty_text_column(df, text_column, name=name)


# ---------------------------------------------------------
# Convert DataFrame to HuggingFace Dataset
# ---------------------------------------------------------


def _to_hf_dataset(df: pd.DataFrame) -> Dataset:
    """
    Convert pandas DataFrame to HuggingFace Dataset.
    """

    dataset = Dataset.from_pandas(df.reset_index(drop=True))

    if "__index_level_0__" in dataset.column_names:
        dataset = dataset.remove_columns(["__index_level_0__"])

    return dataset


# ---------------------------------------------------------
# Training Pipeline
# ---------------------------------------------------------


def train_model(
    df: pd.DataFrame,
    params: dict[str, Any] | None = None,
    text_column: str = "text",
    validation_df: pd.DataFrame | None = None,
    test_df: pd.DataFrame | None = None,
):
    """
    Train RoBERTa fake news classifier.

    Returns
    -------
    Trainer
    Dataset (test)
    """

    try:

        logger.info("Starting RoBERTa training pipeline")

        _validate_split_df(df, "df", text_column)

        if validation_df is not None:
            _validate_split_df(validation_df, "validation_df", text_column)

        if test_df is not None:
            _validate_split_df(test_df, "test_df", text_column)

        torch.manual_seed(SEED)
        np.random.seed(SEED)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info("Training device: %s", device)

        params = params or {}

        learning_rate = float(
            params.get("learning_rate", DEFAULT_LEARNING_RATE)
        )
        batch_size = int(params.get("batch_size", DEFAULT_BATCH_SIZE))
        epochs = int(params.get("epochs", DEFAULT_EPOCHS))
        resume_training = bool(
            params.get(
                "resume_from_checkpoint", DEFAULT_RESUME_FROM_CHECKPOINT
            )
        )

        # Dataset split
        if validation_df is None or test_df is None:
            train_df, val_df, resolved_test_df = _split_train_val_test(df)
        else:
            train_df = df
            val_df = validation_df
            resolved_test_df = test_df

        logger.info(
            "Dataset sizes -> Train:%d Val:%d Test:%d",
            len(train_df),
            len(val_df),
            len(resolved_test_df),
        )

        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        LOGS_DIR.mkdir(parents=True, exist_ok=True)
        MODEL_PATH.mkdir(parents=True, exist_ok=True)

        tokenizer = RobertaTokenizer.from_pretrained(MODEL_NAME)

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

        model = RobertaForSequenceClassification.from_pretrained(
            MODEL_NAME,
            num_labels=2,
            id2label=ID2LABEL,
            label2id=LABEL2ID,
        )

        model.to(device)

        training_args = TrainingArguments(
            output_dir=str(MODELS_DIR),
            learning_rate=learning_rate,
            weight_decay=0.01,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            gradient_accumulation_steps=2,
            num_train_epochs=epochs,
            logging_dir=str(LOGS_DIR),
            logging_steps=100,
            save_strategy="steps",
            save_steps=500,
            evaluation_strategy="steps",
            eval_steps=500,
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

        if resume_training:
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
