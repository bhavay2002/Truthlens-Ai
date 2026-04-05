"""
TruthLens Multi-Task Training Pipeline

Trains a shared RoBERTa encoder with 6 task-specific prediction heads:
  1. Bias detection (binary: non_bias / bias)
  2. Ideology classification (left / center / right)
  3. Propaganda detection (binary: non_propaganda / propaganda)
  4. Narrative role detection (hero / villain / victim — multi-label)
  5. Narrative frame detection (RE / HI / CO / MO / EC — multi-label)
  6. Emotion classification (20-label multi-label)

Entry point: python main.py
Model saved to: models/truthlens_model/
"""

import logging
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer

from src.features.emotion.emotion_schema import EMOTION_LABELS
from src.models.multitask.multitask_truthlens_model import (
    MultiTaskTruthLensConfig,
    MultiTaskTruthLensModel,
)
from src.models.training.trainer import Trainer, TrainerConfig
from src.training.optimizer_factory import create_optimizer
from src.training.scheduler_factory import create_scheduler
from src.utils.config_loader import get_config_value, load_config
from src.utils.logging_utils import configure_logging
from src.utils.settings import load_settings


# -----------------------------------------------------
# Settings
# -----------------------------------------------------

SETTINGS = load_settings()
configure_logging(log_file=SETTINGS.paths.training_log_path)
logger = logging.getLogger(__name__)


# -----------------------------------------------------
# Paths  (resolved from settings / config)
# -----------------------------------------------------

_cfg = load_config()

SPLITS_DIR = Path(
    get_config_value(_cfg, "data", "splits_dir", default="data/splits")
)
TRAIN_PATH = SPLITS_DIR / "train.csv"
VAL_PATH = SPLITS_DIR / "validation.csv"
TEST_PATH = SPLITS_DIR / "test.csv"

MODEL_SAVE_PATH = Path(SETTINGS.model.path)


# -----------------------------------------------------
# Label column definitions
# -----------------------------------------------------

TEXT_COLUMN = get_config_value(
    _cfg, "training", "text_column", default="text"
)

BIAS_LABEL = get_config_value(
    _cfg, "model", "heads", "bias_detection", "label_column",
    default="bias_label",
)
IDEOLOGY_LABEL = get_config_value(
    _cfg, "model", "heads", "ideology_detection", "label_column",
    default="ideology_label",
)
PROPAGANDA_LABEL = get_config_value(
    _cfg, "model", "heads", "propaganda_detection", "label_column",
    default="propaganda_label",
)

NARRATIVE_COLUMNS = ["hero", "villain", "victim"]
FRAME_COLUMNS = ["RE", "HI", "CO", "MO", "EC"]
EMOTION_COLUMNS = [f"emotion_{label}" for label in EMOTION_LABELS]


# -----------------------------------------------------
# Multi-Task Dataset
# -----------------------------------------------------

class TruthLensMultiTaskDataset(Dataset):
    """
    PyTorch Dataset for multi-task TruthLens training.

    Tokenizes article text and assembles a labels dict with one entry
    per task head (only tasks whose columns are present in the DataFrame
    are included in the labels dict).
    """

    def __init__(
        self,
        df: pd.DataFrame,
        tokenizer,
        max_length: int = 256,
        text_column: str = "text",
    ) -> None:
        self.df = df.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.text_column = text_column

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> dict:
        row = self.df.iloc[idx]
        text = str(row[self.text_column])

        encoding = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )

        item: dict = {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
        }

        labels: dict = {}

        if BIAS_LABEL in self.df.columns:
            labels["bias"] = torch.tensor(int(row[BIAS_LABEL]), dtype=torch.long)

        if IDEOLOGY_LABEL in self.df.columns:
            labels["ideology"] = torch.tensor(int(row[IDEOLOGY_LABEL]), dtype=torch.long)

        if PROPAGANDA_LABEL in self.df.columns:
            labels["propaganda"] = torch.tensor(int(row[PROPAGANDA_LABEL]), dtype=torch.long)

        narrative_cols_present = [c for c in NARRATIVE_COLUMNS if c in self.df.columns]
        if narrative_cols_present:
            labels["narrative"] = torch.tensor(
                [float(row[c]) for c in narrative_cols_present],
                dtype=torch.float,
            )

        frame_cols_present = [c for c in FRAME_COLUMNS if c in self.df.columns]
        if frame_cols_present:
            labels["narrative_frame"] = torch.tensor(
                [float(row[c]) for c in frame_cols_present],
                dtype=torch.float,
            )

        emotion_cols_present = [c for c in EMOTION_COLUMNS if c in self.df.columns]
        if emotion_cols_present:
            labels["emotion"] = torch.tensor(
                [float(row[c]) for c in emotion_cols_present],
                dtype=torch.float,
            )

        if labels:
            item["labels"] = labels

        return item


# -----------------------------------------------------
# Load Dataset
# -----------------------------------------------------

def load_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load pre-split train / validation / test CSVs.

    Combines title + text columns when both are present.
    Falls back to a demo dataset if split files do not exist yet.
    """

    for path in (TRAIN_PATH, VAL_PATH, TEST_PATH):
        if not path.exists():
            raise FileNotFoundError(
                f"Split file not found: {path}\n"
                "Run the data pipeline first to generate splits:\n"
                "  Place raw data in data/raw/ then re-run python main.py"
            )

    train_df = pd.read_csv(TRAIN_PATH)
    val_df = pd.read_csv(VAL_PATH)
    test_df = pd.read_csv(TEST_PATH)

    for df in (train_df, val_df, test_df):
        if "title" in df.columns and TEXT_COLUMN in df.columns:
            df[TEXT_COLUMN] = df["title"].fillna("") + " " + df[TEXT_COLUMN].fillna("")
        elif "title" in df.columns and TEXT_COLUMN not in df.columns:
            df[TEXT_COLUMN] = df["title"].fillna("")

    logger.info(
        "Dataset loaded — train: %d  val: %d  test: %d",
        len(train_df), len(val_df), len(test_df),
    )
    return train_df, val_df, test_df


# -----------------------------------------------------
# Save model
# -----------------------------------------------------

def save_model(model: MultiTaskTruthLensModel, tokenizer) -> None:
    """Save model weights and tokenizer to MODEL_SAVE_PATH."""
    MODEL_SAVE_PATH.mkdir(parents=True, exist_ok=True)
    tokenizer.save_pretrained(str(MODEL_SAVE_PATH))
    torch.save(model.state_dict(), MODEL_SAVE_PATH / "pytorch_model.bin")

    import json
    config_data = {
        "model_type": "multitask_truthlens",
        "model_name": model.config.model_name,
        "dropout": model.config.dropout,
        "pooling": model.config.pooling,
        "architectures": ["MultiTaskTruthLensModel"],
        "label2id": {"REAL": 0, "FAKE": 1},
        "id2label": {"0": "REAL", "1": "FAKE"},
    }
    (MODEL_SAVE_PATH / "config.json").write_text(json.dumps(config_data, indent=2))
    logger.info("Model saved to %s", MODEL_SAVE_PATH)


# -----------------------------------------------------
# Main
# -----------------------------------------------------

def main() -> None:

    try:

        logger.info("=== TruthLens Multi-Task Training Pipeline ===")

        # --------------------------------------------------
        # Resolve training hyperparameters from settings
        # --------------------------------------------------

        model_name = SETTINGS.model.name
        max_length = SETTINGS.model.max_length
        epochs = SETTINGS.training.epochs
        batch_size = SETTINGS.training.batch_size
        learning_rate = SETTINGS.training.learning_rate
        seed = SETTINGS.training.seed

        warmup_ratio = float(
            get_config_value(_cfg, "training", "warmup_ratio", default=0.1)
        )
        weight_decay = float(
            get_config_value(_cfg, "training", "weight_decay", default=0.01)
        )
        gradient_accumulation_steps = int(
            get_config_value(_cfg, "training", "gradient_accumulation_steps", default=2)
        )

        torch.manual_seed(seed)
        np.random.seed(seed)

        device_str = SETTINGS.inference.device
        if device_str == "auto":
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            device = torch.device(device_str)

        logger.info("Training device: %s", device)
        logger.info("Model: %s  |  max_length: %d  |  epochs: %d  |  batch: %d  |  lr: %g",
                    model_name, max_length, epochs, batch_size, learning_rate)

        # --------------------------------------------------
        # Load data
        # --------------------------------------------------

        logger.info("Loading datasets from %s", SPLITS_DIR)
        train_df, val_df, _ = load_data()

        # --------------------------------------------------
        # Tokenizer
        # --------------------------------------------------

        logger.info("Loading tokenizer: %s", model_name)
        tokenizer = AutoTokenizer.from_pretrained(model_name)

        # --------------------------------------------------
        # Datasets and DataLoaders
        # --------------------------------------------------

        train_dataset = TruthLensMultiTaskDataset(
            train_df, tokenizer, max_length=max_length, text_column=TEXT_COLUMN
        )
        val_dataset = TruthLensMultiTaskDataset(
            val_df, tokenizer, max_length=max_length, text_column=TEXT_COLUMN
        )

        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
        )

        logger.info(
            "DataLoaders ready — train batches: %d  val batches: %d",
            len(train_loader), len(val_loader),
        )

        # --------------------------------------------------
        # Model
        # --------------------------------------------------

        logger.info("Initializing MultiTaskTruthLensModel")

        model_config = MultiTaskTruthLensConfig(
            model_name=model_name,
            dropout=float(
                get_config_value(_cfg, "model", "architecture", "dropout", default=0.1)
            ),
        )
        model = MultiTaskTruthLensModel(config=model_config)

        # --------------------------------------------------
        # Optimizer and scheduler
        # --------------------------------------------------

        optimizer = create_optimizer(
            model.parameters(),
            optimizer_name="adamw",
            learning_rate=learning_rate,
            weight_decay=weight_decay,
        )

        steps_per_epoch = math.ceil(len(train_dataset) / batch_size)
        total_steps = max(1, math.ceil(steps_per_epoch / gradient_accumulation_steps) * epochs)
        warmup_steps = max(1, int(total_steps * warmup_ratio))

        scheduler = create_scheduler(
            optimizer,
            scheduler_name="linear",
            num_training_steps=total_steps,
            num_warmup_steps=warmup_steps,
        )

        logger.info(
            "Optimizer: AdamW  |  total steps: %d  |  warmup steps: %d",
            total_steps, warmup_steps,
        )

        # --------------------------------------------------
        # Training
        # --------------------------------------------------

        trainer_config = TrainerConfig(
            epochs=epochs,
            gradient_accumulation_steps=gradient_accumulation_steps,
            max_grad_norm=float(
                get_config_value(_cfg, "training", "gradient_clipping", default=1.0)
            ),
            device=str(device),
        )

        trainer = Trainer(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            config=trainer_config,
        )

        logger.info("Starting training for %d epoch(s)", epochs)
        history = trainer.train(train_loader, val_loader)

        logger.info("Training complete")
        for i, (tl, vl) in enumerate(
            zip(history.get("train_loss", []), history.get("val_loss", [])), start=1
        ):
            logger.info("  Epoch %d — train_loss: %.4f  val_loss: %.4f", i, tl, vl)

        # --------------------------------------------------
        # Save
        # --------------------------------------------------

        save_model(model, tokenizer)
        logger.info("=== Pipeline finished successfully ===")

    except FileNotFoundError as e:
        logger.error("Data not found: %s", e)
        sys.exit(1)

    except Exception as e:
        logger.error("Pipeline failed: %s", e, exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
