"""
RoBERTa Training Module for TruthLens AI
Handles dataset splitting, tokenization, training, and model saving
"""

import inspect
import logging

import numpy as np
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

from src.utils.config_loader import get_config_value, get_path, load_config


logger = logging.getLogger(__name__)

CONFIG = load_config()

MODEL_NAME = get_config_value(CONFIG, "model", "name", default="roberta-base")
MAX_LENGTH = int(get_config_value(CONFIG, "model", "max_length", default=256))
SEED = int(get_config_value(CONFIG, "training", "seed", default=42))
DEFAULT_EPOCHS = int(get_config_value(CONFIG, "training", "epochs", default=3))
DEFAULT_BATCH_SIZE = int(get_config_value(CONFIG, "training", "batch_size", default=8))
DEFAULT_LEARNING_RATE = float(
    get_config_value(CONFIG, "training", "learning_rate", default=2e-5)
)

MODELS_DIR = get_path(CONFIG, "paths", "models_dir", default="models")
LOGS_DIR = get_path(CONFIG, "paths", "logs_dir", default="logs")
MODEL_PATH = get_path(CONFIG, "model", "path", default="models/roberta_model")
TEST_SET_PATH = get_path(
    CONFIG,
    "data",
    "test_set_path",
    default="data/processed/test_set.csv",
)

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
# Training Function
# -------------------------------------------------

def train_model(df, params=None, text_column: str = "text"):
    try:
        if text_column not in df.columns:
            raise ValueError(f"Column '{text_column}' not found in dataframe")

        logger.info("Starting model training...")

        torch.manual_seed(SEED)
        np.random.seed(SEED)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info("Using device: %s", device)

        # -------------------------------------------------
        # Hyperparameters from config / Optuna overrides
        # -------------------------------------------------

        learning_rate = (
            params.get("learning_rate", DEFAULT_LEARNING_RATE)
            if params
            else DEFAULT_LEARNING_RATE
        )
        batch_size = (
            params.get("batch_size", DEFAULT_BATCH_SIZE)
            if params
            else DEFAULT_BATCH_SIZE
        )
        epochs = params.get("epochs", DEFAULT_EPOCHS) if params else DEFAULT_EPOCHS

        logger.info(
            "Training configuration -> LR: %s, Batch Size: %s, Epochs: %s, Text Column: %s",
            learning_rate,
            batch_size,
            epochs,
            text_column,
        )

        # -------------------------------------------------
        # Train / Validation / Test Split
        # -------------------------------------------------

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

        # -------------------------------------------------
        # Tokenizer
        # -------------------------------------------------

        tokenizer = RobertaTokenizer.from_pretrained(MODEL_NAME)

        # -------------------------------------------------
        # Convert to HuggingFace Dataset
        # -------------------------------------------------

        train_dataset = Dataset.from_pandas(train_df)
        val_dataset = Dataset.from_pandas(val_df)
        test_dataset = Dataset.from_pandas(test_df)

        # Remove pandas index artifact column
        for dataset_name, dataset in (
            ("train", train_dataset),
            ("val", val_dataset),
            ("test", test_dataset),
        ):
            if "__index_level_0__" in dataset.column_names:
                cleaned = dataset.remove_columns(["__index_level_0__"])
                if dataset_name == "train":
                    train_dataset = cleaned
                elif dataset_name == "val":
                    val_dataset = cleaned
                else:
                    test_dataset = cleaned

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

        # -------------------------------------------------
        # Model
        # -------------------------------------------------

        model = RobertaForSequenceClassification.from_pretrained(
            MODEL_NAME,
            num_labels=2,
            id2label=ID2LABEL,
            label2id=LABEL2ID,
        )

        logger.info("Using label mapping: %s", model.config.id2label)

        model.to(device)

        # -------------------------------------------------
        # Training Arguments
        # -------------------------------------------------

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
            "dataloader_num_workers": 4,
            "dataloader_pin_memory": True,
        }

        # Compatibility with transformers versions
        training_args_signature = inspect.signature(TrainingArguments.__init__)

        if "eval_strategy" in training_args_signature.parameters:
            training_kwargs["eval_strategy"] = "epoch"
        else:
            training_kwargs["evaluation_strategy"] = "epoch"

        training_args = TrainingArguments(**training_kwargs)

        # -------------------------------------------------
        # Trainer
        # -------------------------------------------------

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            compute_metrics=compute_metrics,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
        )

        # -------------------------------------------------
        # Training
        # -------------------------------------------------

        logger.info("Training model...")
        trainer.train()

        # -------------------------------------------------
        # Save Model
        # -------------------------------------------------

        logger.info("Saving model...")

        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        LOGS_DIR.mkdir(parents=True, exist_ok=True)
        MODEL_PATH.mkdir(parents=True, exist_ok=True)
        TEST_SET_PATH.parent.mkdir(parents=True, exist_ok=True)

        model.save_pretrained(str(MODEL_PATH))
        tokenizer.save_pretrained(str(MODEL_PATH))

        test_df.to_csv(TEST_SET_PATH, index=False)

        logger.info("Training complete!")

        return trainer, test_dataset

    except Exception as e:
        logger.error("Error during training: %s", e)
        raise
