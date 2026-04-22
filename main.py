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

import hashlib
import logging
import os
import math
import shutil
import sys
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer

from src.evaluation.evaluate_model import evaluate
from src.evaluation.report_writer import save_report
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

# m1: tokenizers + DataLoader workers can deadlock; disable parallelism in tokenizer.
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


# -----------------------------------------------------
# Config
# -----------------------------------------------------

_cfg = load_config()


# -----------------------------------------------------
# Data + save paths (env-overridable; m2)
# -----------------------------------------------------

_DEFAULT_DRIVE_DATA = Path("/content/drive/MyDrive/truthlens unified data")
DRIVE_DATA_PATH = Path(os.environ.get("TRUTHLENS_DATA_DIR", str(_DEFAULT_DRIVE_DATA)))

TRAIN_PATH = DRIVE_DATA_PATH / "unified_dataset_train.csv"
VAL_PATH = DRIVE_DATA_PATH / "unified_dataset_validation.csv"
TEST_PATH = DRIVE_DATA_PATH / "unified_dataset_test.csv"

_DEFAULT_LOCAL_SAVE = Path("/content/truthlens_model")
LOCAL_SAVE_PATH = Path(os.environ.get("TRUTHLENS_LOCAL_SAVE", str(_DEFAULT_LOCAL_SAVE)))
DRIVE_SAVE_PATH = Path(os.environ.get("TRUTHLENS_DRIVE_SAVE", str(SETTINGS.model.path)))


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
# Save Model — synchronous, atomic, ordered, verified  (C1, C6, C7)
# -----------------------------------------------------

def _md5(p: Path) -> str:
    h = hashlib.md5()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _sync_to_drive(src: Path, dst: Path, retries: int = 3) -> None:
    create_folder(dst)
    for f in src.iterdir():
        if not f.is_file():
            continue
        target = dst / f.name
        last_exc: Exception | None = None
        for attempt in range(1, retries + 1):
            try:
                tmp = target.with_suffix(target.suffix + ".tmp")
                shutil.copy2(f, tmp)
                os.replace(tmp, target)
                if target.stat().st_size != f.stat().st_size:
                    raise IOError(f"Size mismatch after copy: {target}")
                if _md5(target) != _md5(f):
                    raise IOError(f"Checksum mismatch after copy: {target}")
                last_exc = None
                break
            except Exception as exc:
                last_exc = exc
                logger.warning(
                    "Drive copy attempt %d/%d failed for %s: %s",
                    attempt, retries, f.name, exc,
                )
        if last_exc is not None:
            raise RuntimeError(f"Drive sync failed for {f.name}") from last_exc


def save_model(model, tokenizer):

    create_folder(LOCAL_SAVE_PATH)

    # NaN/Inf guard before serialization (C6)
    raw_model = getattr(model, "_orig_mod", model)
    state = {k: v.detach().cpu() for k, v in raw_model.state_dict().items()}
    for k, v in state.items():
        if torch.is_tensor(v) and v.is_floating_point() and not torch.isfinite(v).all():
            raise RuntimeError(f"Refusing to save: non-finite values in {k}")

    # Atomic write of model weights
    final = LOCAL_SAVE_PATH / "pytorch_model.bin"
    tmp = LOCAL_SAVE_PATH / "pytorch_model.bin.tmp"
    torch.save(state, tmp)
    os.replace(tmp, final)

    # Tokenizer + config (synchronous so Drive sync sees a complete tree)
    tokenizer.save_pretrained(str(LOCAL_SAVE_PATH))
    save_json(
        {
            "model_type": "multitask_truthlens",
            "architectures": ["MultiTaskTruthLensModel"],
        },
        LOCAL_SAVE_PATH / "config.json",
        indent=2,
    )

    logger.info("Local save complete: %s", final)

    # Drive sync only AFTER local save is durable
    if DRIVE_SAVE_PATH.parent.exists() or DRIVE_SAVE_PATH.exists():
        try:
            _sync_to_drive(LOCAL_SAVE_PATH, DRIVE_SAVE_PATH)
            logger.info("Drive sync complete: %s", DRIVE_SAVE_PATH)
        except Exception as exc:
            logger.error("Drive sync failed: %s", exc)
            raise


# -----------------------------------------------------
# Final Test Evaluation (M2)
# -----------------------------------------------------

def _evaluate_on_test(model, test_loader, device) -> None:
    model.eval()
    y_true: list[int] = []
    y_pred: list[int] = []
    y_proba: list[float] = []

    raw = getattr(model, "_orig_mod", model)
    with torch.no_grad():
        for batch in test_loader:
            inputs = {
                k: (v.to(device) if isinstance(v, torch.Tensor) else v)
                for k, v in batch.items() if k != "labels"
            }
            labels = batch.get("labels", {})
            outputs = raw(**inputs)

            logits = None
            if isinstance(outputs, dict):
                heads = outputs.get("heads") or outputs.get("logits") or {}
                if isinstance(heads, dict):
                    logits = heads.get("bias")
                elif torch.is_tensor(heads):
                    logits = heads
            if logits is None:
                continue

            probs = torch.softmax(logits.float(), dim=-1)
            preds = probs.argmax(dim=-1)

            y_true.extend(labels["bias"].cpu().tolist())
            y_pred.extend(preds.cpu().tolist())
            y_proba.extend(probs[:, 1].cpu().tolist() if probs.shape[-1] > 1 else probs.squeeze(-1).cpu().tolist())

    if not y_true:
        logger.warning("Test evaluation skipped: no bias logits returned by model")
        return

    summary = evaluate(y_true, y_pred, y_proba)
    report = {"summary": summary, "tasks": {"bias": summary}}
    out = SETTINGS.paths.evaluation_results_path
    save_report(report, out, generate_plots=False)
    logger.info("Test report saved: %s", out)


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

        # C8: cudnn.benchmark only meaningful on CUDA + cuDNN
        if torch.cuda.is_available() and torch.backends.cudnn.is_available():
            torch.backends.cudnn.benchmark = True

        # M1: gate pin_memory on CUDA availability
        _pin = torch.cuda.is_available()

        # --------------------------------------------------
        # Data
        # --------------------------------------------------

        train_df, val_df, test_df = load_data()

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
                values = [item["labels"][key] for item in batch]
                if isinstance(values[0], (list, tuple)):
                    batched_labels[key] = torch.tensor(values, dtype=torch.float)
                elif isinstance(values[0], float):
                    batched_labels[key] = torch.tensor(values, dtype=torch.float)
                else:
                    batched_labels[key] = torch.tensor(values, dtype=torch.long)

            enc["labels"] = batched_labels
            if _pin:
                return {
                    key: (value.pin_memory() if isinstance(value, torch.Tensor) else value)
                    for key, value in enc.items()
                }
            return dict(enc)

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

        test_dataset = TruthLensMultiTaskDataset(
            test_df,
            max_length=max_length,
            text_column=TEXT_COLUMN,
        )

        _num_workers = 4 if _pin else 0
        _persistent = bool(_num_workers)

        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=collate_fn,
            num_workers=_num_workers,
            pin_memory=_pin,
            persistent_workers=_persistent,
            prefetch_factor=2 if _persistent else None,
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            collate_fn=collate_fn,
            num_workers=_num_workers,
            pin_memory=_pin,
            persistent_workers=_persistent,
            prefetch_factor=2 if _persistent else None,
        )

        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            collate_fn=collate_fn,
            num_workers=_num_workers,
            pin_memory=_pin,
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

        # C2: torch.compile is owned by Trainer.__init__ — do not double-compile.

        # --------------------------------------------------
        # Optimizer
        # --------------------------------------------------

        try:
            optimizer = create_optimizer(
                model,
                optimizer_name="adamw",
                learning_rate=learning_rate,
                weight_decay=weight_decay,
            )
        except ValueError as e:
            logger.warning(f"{e} → Falling back to AdamW")
            optimizer = create_optimizer(
                model,
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
        # Trainer  (C3: wire checkpoint_dir; gate AMP on CUDA)
        # --------------------------------------------------

        trainer_config = TrainerConfig(
            epochs=epochs,
            gradient_accumulation_steps=gradient_accumulation_steps,
            device=str(device),
            use_amp=(device.type == "cuda"),
            amp_dtype="bf16",
            checkpoint_dir=str(SETTINGS.paths.models_dir / "checkpoints"),
            checkpoint_every_steps=0,  # epoch-based saves handled in Trainer.train
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
        # Save + final test evaluation (M2)
        # --------------------------------------------------

        save_model(trainer.model, tokenizer)

        try:
            _evaluate_on_test(trainer.model, test_loader, device)
        except Exception as exc:
            logger.error("Final test evaluation failed: %s", exc, exc_info=True)

        logger.info("Pipeline finished | history=%s", history)

    except Exception as e:

        logger.error("Training failed: %s", e, exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
