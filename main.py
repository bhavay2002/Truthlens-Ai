"""
TruthLens Multi-Task Training Pipeline

Trains a shared RoBERTa encoder with 6 task-specific prediction heads:
  1. Bias detection
  2. Ideology classification
  3. Propaganda detection
  4. Narrative role detection
  5. Narrative frame detection
  6. Emotion classification
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
from src.utils.helper_functions import create_folder
from src.utils.json_utils import save_json
from src.utils.logging_utils import configure_logging
from src.utils.seed_utils import set_seed
from src.utils.settings import load_settings
from src.utils.device_utils import get_device


# -----------------------------------------------------
# Settings
# -----------------------------------------------------

SETTINGS = load_settings()
configure_logging(log_file=SETTINGS.paths.training_log_path)
logger = logging.getLogger(__name__)


# -----------------------------------------------------
# Config
# -----------------------------------------------------

_cfg = load_config()


# -----------------------------------------------------
# Google Drive dataset paths
# -----------------------------------------------------

DRIVE_DATA_PATH = Path("/content/drive/MyDrive/truthlens unified data")

TRAIN_PATH = DRIVE_DATA_PATH / "unified_dataset_train.csv"
VAL_PATH = DRIVE_DATA_PATH / "unified_dataset_validation.csv"
TEST_PATH = DRIVE_DATA_PATH / "unified_dataset_test.csv"

MODEL_SAVE_PATH = Path(SETTINGS.model.path)


# -----------------------------------------------------
# Label columns
# -----------------------------------------------------

TEXT_COLUMN = get_config_value(_cfg, "training", "text_column", default="text")

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
EMOTION_COLUMNS = [f"emotion_{i}" for i in range(len(EMOTION_LABELS))]


# -----------------------------------------------------
# Dataset
# -----------------------------------------------------
class TruthLensMultiTaskDataset(Dataset):

    def __init__(self, df, tokenizer, max_length=256, text_column="text"):
        self.df = df.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.text_column = text_column

    def __len__(self):
        return len(self.df)

    def _safe_int(self, value, default=0):
        try:
            if pd.isna(value):
                return default
            return int(float(value))
        except Exception:
            return default

    def _safe_float(self, value, default=0.0):
        try:
            if pd.isna(value):
                return default
            return float(value)
        except Exception:
            return default

    def _entity_to_binary(self, entity_field) -> int:
        if entity_field is None:
            return 0
        if isinstance(entity_field, str) and entity_field.strip() == "":
            return 0
        try:
            if pd.isna(entity_field):
                return 0
        except Exception:
            pass
        return 1

    def __getitem__(self, idx):

        row = self.df.iloc[idx]
        text = str(row[self.text_column])

        encoding = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )

        item = {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
        }

        labels = {
            "bias": torch.tensor(
                self._safe_int(row.get(BIAS_LABEL, 0)),
                dtype=torch.long
            ),
            "ideology": torch.tensor(
                self._safe_int(row.get(IDEOLOGY_LABEL, 1)),
                dtype=torch.long
            ),
            "propaganda": torch.tensor(
                self._safe_int(row.get(PROPAGANDA_LABEL, 0)),
                dtype=torch.long
            ),
            "narrative": torch.tensor(
                [
                    float(max(
                        self._safe_int(row.get("hero", 0)),
                        self._entity_to_binary(row.get("hero_entities")),
                    )),
                    float(max(
                        self._safe_int(row.get("villain", 0)),
                        self._entity_to_binary(row.get("villain_entities")),
                    )),
                    float(max(
                        self._safe_int(row.get("victim", 0)),
                        self._entity_to_binary(row.get("victim_entities")),
                    )),
                ],
                dtype=torch.float,
            ),
            "narrative_frame": torch.tensor(
                [self._safe_float(row.get(c, 0)) for c in FRAME_COLUMNS],
                dtype=torch.float,
            ),
            "emotion": torch.tensor(
                [self._safe_float(row.get(c, 0)) for c in EMOTION_COLUMNS],
                dtype=torch.float,
            ),
        }

        item["labels"] = labels

        item["hero_entities"] = str(row.get("hero_entities") or "")
        item["villain_entities"] = str(row.get("villain_entities") or "")
        item["victim_entities"] = str(row.get("victim_entities") or "")

        return item

# -----------------------------------------------------
# Load Data
# -----------------------------------------------------

def load_data():

    train_df = pd.read_csv(TRAIN_PATH)
    val_df = pd.read_csv(VAL_PATH)
    test_df = pd.read_csv(TEST_PATH)

    for df in (train_df, val_df, test_df):
        if "title" in df.columns and TEXT_COLUMN in df.columns:
            df[TEXT_COLUMN] = df["title"].fillna("") + " " + df[TEXT_COLUMN].fillna("")

    logger.info(
        "Dataset loaded — train: %d  val: %d  test: %d",
        len(train_df),
        len(val_df),
        len(test_df),
    )

    return train_df, val_df, test_df


# -----------------------------------------------------
# Save Model
# -----------------------------------------------------

def save_model(model, tokenizer):

    create_folder(MODEL_SAVE_PATH)

    tokenizer.save_pretrained(str(MODEL_SAVE_PATH))

    torch.save(
        model.state_dict(),
        MODEL_SAVE_PATH / "pytorch_model.bin"
    )

    config_data = {
        "model_type": "multitask_truthlens",
        "architectures": ["MultiTaskTruthLensModel"],
    }

    save_json(
        config_data,
        MODEL_SAVE_PATH / "config.json",
        indent=2
    )

    logger.info("Model saved to %s", MODEL_SAVE_PATH)


# -----------------------------------------------------
# Main
# -----------------------------------------------------

def main():

    try:

        logger.info("=== TruthLens Multi-Task Training Pipeline ===")

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

        set_seed(seed)

        device = get_device(prefer_gpu=True)

        logger.info("Training device: %s", device)

        # --------------------------------------------------
        # Data
        # --------------------------------------------------

        train_df, val_df, _ = load_data()

        tokenizer = AutoTokenizer.from_pretrained(model_name)

        train_dataset = TruthLensMultiTaskDataset(
            train_df,
            tokenizer,
            max_length=max_length,
            text_column=TEXT_COLUMN,
        )

        val_dataset = TruthLensMultiTaskDataset(
            val_df,
            tokenizer,
            max_length=max_length,
            text_column=TEXT_COLUMN,
        )

        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
        )

        # --------------------------------------------------
        # Model
        # --------------------------------------------------

        model_config = MultiTaskTruthLensConfig(model_name=model_name)

        model = MultiTaskTruthLensModel(config=model_config)

        # --------------------------------------------------
        # Optimizer
        # --------------------------------------------------

        optimizer = create_optimizer(
            model.parameters(),
            optimizer_name="adamw",
            learning_rate=learning_rate,
            weight_decay=weight_decay,
        )

        steps_per_epoch = math.ceil(len(train_dataset) / batch_size)

        total_steps = max(
            1,
            math.ceil(steps_per_epoch / gradient_accumulation_steps) * epochs,
        )

        warmup_steps = int(total_steps * warmup_ratio)

        scheduler = create_scheduler(
            optimizer,
            scheduler_name="linear",
            num_training_steps=total_steps,
            num_warmup_steps=warmup_steps,
        )

        # --------------------------------------------------
        # Trainer
        # --------------------------------------------------

        trainer_config = TrainerConfig(
            epochs=epochs,
            gradient_accumulation_steps=gradient_accumulation_steps,
            device=str(device),
        )

        trainer = Trainer(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            config=trainer_config,
        )

        logger.info("Starting training")

        history = trainer.train(train_loader, val_loader)

        logger.info("Training complete")

        # --------------------------------------------------
        # Save
        # --------------------------------------------------

        save_model(model, tokenizer)

        logger.info("Pipeline finished")

    except Exception as e:

        logger.error("Training failed: %s", e, exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()