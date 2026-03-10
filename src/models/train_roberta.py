"""
RoBERTa Training Module for TruthLens AI
Handles dataset splitting, tokenization, training, and model saving
"""

import torch
import numpy as np
import logging
import inspect
from datasets import Dataset
from transformers import (
    RobertaTokenizer,
    RobertaForSequenceClassification,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback
)
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MODEL_NAME = "roberta-base"
MAX_LENGTH = 256
SEED = 42
ID2LABEL = {0: "REAL", 1: "FAKE"}
LABEL2ID = {"REAL": 0, "FAKE": 1}


# -------------------------------------------------
# Evaluation Metrics
# -------------------------------------------------

def compute_metrics(eval_pred):
    """
    Compute evaluation metrics
    """

    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)

    precision, recall, f1, _ = precision_recall_fscore_support(
        labels,
        preds,
        average="binary"
    )

    acc = accuracy_score(labels, preds)

    try:
        probs = torch.softmax(torch.tensor(logits), dim=1)[:, 1].numpy()
        roc_auc = roc_auc_score(labels, probs)
    except:
        roc_auc = 0.0

    return {
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "roc_auc": roc_auc
    }


# -------------------------------------------------
# Tokenization Function
# -------------------------------------------------

def tokenize_function(example, tokenizer):
    return tokenizer(
        example["text"],
        truncation=True,
        padding="max_length",
        max_length=MAX_LENGTH
    )


# -------------------------------------------------
# Training Function
# -------------------------------------------------

def train_model(df):
    """
    Train RoBERTa model with proper train/validation/test splits
    """

    try:

        logger.info("Starting model training...")

        torch.manual_seed(SEED)
        np.random.seed(SEED)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {device}")

        # -------------------------------------------------
        # Train / Validation / Test Split
        # -------------------------------------------------

        train_df, temp_df = train_test_split(
            df,
            test_size=0.3,
            random_state=SEED,
            stratify=df["label"]
        )

        val_df, test_df = train_test_split(
            temp_df,
            test_size=0.5,
            random_state=SEED,
            stratify=temp_df["label"]
        )

        logger.info(
            f"Train samples: {len(train_df)}, "
            f"Val samples: {len(val_df)}, "
            f"Test samples: {len(test_df)}"
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

        # Remove pandas index artifact column if present
        if "__index_level_0__" in train_dataset.column_names:
            train_dataset = train_dataset.remove_columns(["__index_level_0__"])
        if "__index_level_0__" in val_dataset.column_names:
            val_dataset = val_dataset.remove_columns(["__index_level_0__"])
        if "__index_level_0__" in test_dataset.column_names:
            test_dataset = test_dataset.remove_columns(["__index_level_0__"])

        train_dataset = train_dataset.map(
            lambda x: tokenize_function(x, tokenizer),
            batched=True
        )

        val_dataset = val_dataset.map(
            lambda x: tokenize_function(x, tokenizer),
            batched=True
        )

        test_dataset = test_dataset.map(
            lambda x: tokenize_function(x, tokenizer),
            batched=True
        )

        train_dataset.set_format(
            type="torch",
            columns=["input_ids", "attention_mask", "label"]
        )

        val_dataset.set_format(
            type="torch",
            columns=["input_ids", "attention_mask", "label"]
        )

        test_dataset.set_format(
            type="torch",
            columns=["input_ids", "attention_mask", "label"]
        )

        # -------------------------------------------------
        # Model
        # -------------------------------------------------

        model = RobertaForSequenceClassification.from_pretrained(
            MODEL_NAME,
            num_labels=2,
            id2label=ID2LABEL,
            label2id=LABEL2ID
        )

        logger.info(f"Using label mapping: {model.config.id2label}")

        model.to(device)

        # -------------------------------------------------
        # Training Arguments
        # -------------------------------------------------

        training_kwargs = {
            "output_dir": "./models",
            "learning_rate": 2e-5,
            "weight_decay": 0.01,
            "per_device_train_batch_size": 8,
            "per_device_eval_batch_size": 8,
            "gradient_accumulation_steps": 2,
            "num_train_epochs": 3,
            "save_strategy": "epoch",
            "logging_dir": "./logs",
            "logging_steps": 100,
            "load_best_model_at_end": True,
            "metric_for_best_model": "f1",
            "save_total_limit": 2,
            "fp16": torch.cuda.is_available(),
            "seed": SEED,
        }

        # transformers>=5 uses eval_strategy; older versions use evaluation_strategy
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
            callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
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

        model.save_pretrained("./models/roberta_model")
        tokenizer.save_pretrained("./models/roberta_model")

        # Save test set for evaluation
        test_df.to_csv("./data/processed/test_set.csv", index=False)

        logger.info("Training complete!")

        return trainer, test_dataset

    except Exception as e:
        logger.error(f"Error during training: {e}")
        raise
