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
import math
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
UNIFIED_LABEL_CANDIDATES = (
    "bias_label",
    "ideology_label",
    "propaganda_label",
    "frame",
)


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
    *,
    label_column: str = "label",
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


# ---------------------------------------------------------
# Dataset Validation
# ---------------------------------------------------------


def _validate_split_df(
    df: pd.DataFrame,
    name: str,
    text_column: str,
    label_column: str = "label",
):
    """
    Validate dataframe integrity.
    """

    ensure_dataframe(df, name=name, required_columns=[text_column, label_column], min_rows=2)
    ensure_non_empty_text_column(df, text_column, name=name)


def _resolve_label_column(df: pd.DataFrame, label_column: str) -> str:
    if label_column in df.columns:
        return label_column

    if label_column != "label":
        raise ValueError(
            f"label column '{label_column}' not found in dataframe columns."
        )

    for candidate in UNIFIED_LABEL_CANDIDATES:
        if candidate in df.columns:
            logger.info(
                "Column 'label' not found. Using '%s' as label column.",
                candidate,
            )
            return candidate

    raise ValueError(
        "No usable label column found. Expected 'label' or one of "
        f"{list(UNIFIED_LABEL_CANDIDATES)}."
    )


def _build_label_mapping(
    labels: pd.Series,
) -> tuple[dict[Any, int], dict[int, str], dict[str, int]]:
    unique_labels = labels.dropna().tolist()
    unique_labels = list(dict.fromkeys(unique_labels))

    if not unique_labels:
        raise ValueError("Cannot build label mapping from empty label series.")

    # Keep stable ordering but avoid mixed-type sort failures.
    unique_labels = sorted(unique_labels, key=lambda value: str(value))

    label_to_id = {label: idx for idx, label in enumerate(unique_labels)}
    id2label = {idx: str(label).upper() for label, idx in label_to_id.items()}
    label2id = {name: idx for idx, name in id2label.items()}

    return label_to_id, id2label, label2id


def _attach_internal_label(
    df: pd.DataFrame,
    *,
    source_label_column: str,
    label_to_id: dict[Any, int],
) -> pd.DataFrame:
    prepared = df.copy()
    prepared["label"] = prepared[source_label_column].map(label_to_id)

    if prepared["label"].isna().any():
        raise ValueError(
            f"Encountered unknown labels in column '{source_label_column}' "
            "when building internal training labels."
        )

    prepared["label"] = prepared["label"].astype(int)
    return prepared


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


def _compute_checkpoint_save_steps(
    *,
    train_examples: int,
    batch_size: int,
    gradient_accumulation_steps: int,
    epochs: int,
) -> int:
    """
    Compute checkpoint frequency so a save occurs every 10% of training.
    """

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
    total_optimizer_steps = max(1, optimizer_steps_per_epoch * epochs)

    return max(1, math.ceil(total_optimizer_steps * 0.10))


# ---------------------------------------------------------
# Training Pipeline
# ---------------------------------------------------------


def train_model(
    df: pd.DataFrame,
    params: dict[str, Any] | None = None,
    text_column: str = "text",
    label_column: str = "label",
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

        resolved_label_column = _resolve_label_column(df, label_column)

        _validate_split_df(df, "df", text_column, resolved_label_column)

        if validation_df is not None:
            _validate_split_df(
                validation_df,
                "validation_df",
                text_column,
                resolved_label_column,
            )

        if test_df is not None:
            _validate_split_df(
                test_df,
                "test_df",
                text_column,
                resolved_label_column,
            )

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
        gradient_accumulation_steps = 2
        resume_training = bool(
            params.get(
                "resume_from_checkpoint", DEFAULT_RESUME_FROM_CHECKPOINT
            )
        )

        # Dataset split
        if validation_df is None or test_df is None:
            train_df, val_df, resolved_test_df = _split_train_val_test(
                df,
                label_column=resolved_label_column,
            )
        else:
            train_df = df
            val_df = validation_df
            resolved_test_df = test_df

        label_to_id, id2label, label2id = _build_label_mapping(
            train_df[resolved_label_column]
        )
        num_labels = len(label_to_id)

        train_df = _attach_internal_label(
            train_df,
            source_label_column=resolved_label_column,
            label_to_id=label_to_id,
        )
        val_df = _attach_internal_label(
            val_df,
            source_label_column=resolved_label_column,
            label_to_id=label_to_id,
        )
        resolved_test_df = _attach_internal_label(
            resolved_test_df,
            source_label_column=resolved_label_column,
            label_to_id=label_to_id,
        )

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
            num_labels=num_labels,
            id2label=id2label,
            label2id=label2id,
        )

        model.to(device)

        checkpoint_save_steps = _compute_checkpoint_save_steps(
            train_examples=len(train_df),
            batch_size=batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            epochs=epochs,
        )

        logger.info(
            "Checkpoint cadence configured at every 10%% of training (%d steps).",
            checkpoint_save_steps,
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
