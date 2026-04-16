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
import os
import math
import shutil
import sys
import threading
from pathlib import Path

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
os.environ["TOKENIZERS_PARALLELISM"] = "true"


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

LOCAL_SAVE_PATH = Path("/content/truthlens_model")
DRIVE_SAVE_PATH = Path(SETTINGS.model.path)


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

    def __init__(self, df, max_length=256, text_column="text"):
        self.df = df.reset_index(drop=True)
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

        labels = {
            "bias": self._safe_int(row.get(BIAS_LABEL, 0)),
            "ideology": self._safe_int(row.get(IDEOLOGY_LABEL, 1)),
            "propaganda": self._safe_int(row.get(PROPAGANDA_LABEL, 0)),
            "narrative": [
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
            "narrative_frame": [self._safe_float(row.get(c, 0)) for c in FRAME_COLUMNS],
            "emotion": [self._safe_float(row.get(c, 0)) for c in EMOTION_COLUMNS],
        }

        return {
            "text": text,
            "labels": labels,
            "hero_entities": row.get("hero_entities", ""),
            "villain_entities": row.get("villain_entities", ""),
            "victim_entities": row.get("victim_entities", ""),
        }

# -----------------------------------------------------
# Load Data
# -----------------------------------------------------

def load_data():

    train_df = pd.read_csv(TRAIN_PATH)
    val_df = pd.read_csv(VAL_PATH)
    test_df = pd.read_csv(TEST_PATH)

    for df in (train_df, val_df, test_df):
        if "title" in df.columns and TEXT_COLUMN in df.columns:
            df[TEXT_COLUMN] = df["title"].fillna("").str.cat(
                df[TEXT_COLUMN].fillna(""),
                sep=" ",
            )

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

    create_folder(LOCAL_SAVE_PATH)

    tokenizer.save_pretrained(str(LOCAL_SAVE_PATH))

    def _save_local(m):
        torch.save(
            m.state_dict(),
            LOCAL_SAVE_PATH / "pytorch_model.bin"
        )

    save_thread = threading.Thread(target=_save_local, args=(model,), daemon=True)
    save_thread.start()

    def _copy_to_drive():
        create_folder(DRIVE_SAVE_PATH)
        for file in LOCAL_SAVE_PATH.iterdir():
            shutil.copy2(file, DRIVE_SAVE_PATH / file.name)

    copy_thread = threading.Thread(target=_copy_to_drive, daemon=True)
    copy_thread.start()

    config_data = {
        "model_type": "multitask_truthlens",
        "architectures": ["MultiTaskTruthLensModel"],
    }

    save_json(
        config_data,
        LOCAL_SAVE_PATH / "config.json",
        indent=2
    )

    save_thread.join(timeout=10)
    copy_thread.join(timeout=10)

    logger.info("Model saved locally and copying to Drive async")


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

        torch.backends.cudnn.benchmark = True

        # --------------------------------------------------
        # Data
        # --------------------------------------------------

        train_df, val_df, _ = load_data()

        tokenizer = AutoTokenizer.from_pretrained(model_name)
        tok = tokenizer

        def collate_fn(batch):
            texts = [item["text"] for item in batch]
            enc = tok(
                texts,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )

            batched_labels = {}
            label_keys = list(batch[0]["labels"])
            for key in label_keys:
                get_labels = lambda item: item["labels"][key]
                values = list(map(get_labels, batch))
                if isinstance(values[0], (list, tuple)):
                    batched_labels[key] = torch.tensor(values, dtype=torch.float)
                elif isinstance(values[0], float):
                    batched_labels[key] = torch.tensor(values, dtype=torch.float)
                else:
                    batched_labels[key] = torch.tensor(values, dtype=torch.long)

            enc["labels"] = batched_labels
            return {
                key: value.pin_memory() if isinstance(value, torch.Tensor) else value
                for key, value in enc.items()
            }

        train_dataset = TruthLensMultiTaskDataset(
            train_df,
            max_length=max_length,
            text_column=TEXT_COLUMN,
        )

        val_dataset = TruthLensMultiTaskDataset(
            val_df,
            max_length=max_length,
            text_column=TEXT_COLUMN,
        )

        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=collate_fn,
            num_workers=4,
            pin_memory=True,
            persistent_workers=True,
            prefetch_factor=2,
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            collate_fn=collate_fn,
            num_workers=4,
            pin_memory=True,
            persistent_workers=True,
            prefetch_factor=2,
        )

        # --------------------------------------------------
        # Model
        # --------------------------------------------------

        model_config = MultiTaskTruthLensConfig(model_name=model_name)

        model = MultiTaskTruthLensModel(config=model_config)
        model = model.to(device)

        if hasattr(model, "encoder") and hasattr(model.encoder, "gradient_checkpointing_enable"):
            model.encoder.gradient_checkpointing_enable()

        if hasattr(model, "config") and hasattr(model.config, "use_flash_attention"):
            model.config.use_flash_attention = True

        if hasattr(torch, "compile"):
            model = torch.compile(model, mode="max-autotune")

        # --------------------------------------------------
        # Optimizer
        # --------------------------------------------------

        try:
            optimizer = create_optimizer(
                model.parameters(),
                optimizer_name="adamw_fused",
                learning_rate=learning_rate,
                weight_decay=weight_decay,
            )
        except Exception:
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
            use_amp=True,
            amp_dtype="bf16",
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