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
import random
import shutil
import signal
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset, Sampler
from transformers import AutoTokenizer, DataCollatorWithPadding

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

# Candidate dataset locations, checked in order. The first directory that
# contains ``unified_dataset_train.csv`` wins. Override unconditionally
# with ``TRUTHLENS_DATA_DIR=/some/path``.
_DATA_CANDIDATES = [
    # Lightning AI Studios working dir
    Path("/teamspace/studios/this_studio/data"),
    Path("/teamspace/studios/this_studio/Truthlens-Ai/data"),
    # Repo-local
    Path(__file__).resolve().parent / "data",
    # Google Colab + Drive mount
    Path("/content/drive/MyDrive/truthlens unified data"),
    Path("/content/data"),
]

_TRAIN_FILE = "unified_dataset_train.csv"


def _resolve_data_dir() -> Path:
    env = os.environ.get("TRUTHLENS_DATA_DIR")
    if env:
        return Path(env).expanduser()
    for cand in _DATA_CANDIDATES:
        if (cand / _TRAIN_FILE).is_file():
            return cand
    # Nothing found — return the first repo-local candidate so the
    # FileNotFoundError below is actionable (path printed in the error).
    return Path(__file__).resolve().parent / "data"


DRIVE_DATA_PATH = _resolve_data_dir()

TRAIN_PATH = DRIVE_DATA_PATH / _TRAIN_FILE
VAL_PATH = DRIVE_DATA_PATH / "unified_dataset_validation.csv"
TEST_PATH = DRIVE_DATA_PATH / "unified_dataset_test.csv"

# Local save dir: prefer a writable, platform-appropriate default.
if Path("/teamspace/studios/this_studio").is_dir():
    _DEFAULT_LOCAL_SAVE = Path("/teamspace/studios/this_studio/truthlens_model")
elif Path("/content").is_dir():
    _DEFAULT_LOCAL_SAVE = Path("/content/truthlens_model")
else:
    _DEFAULT_LOCAL_SAVE = Path(__file__).resolve().parent / "artifacts" / "model"

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
# Dataset (pre-tokenized, tensor-ready, length-aware)
# -----------------------------------------------------
def _check_label_leakage(train_df, val_df, text_column: str, sample: int = 10000) -> None:
    """Warn if train/val share identical input texts (label-leakage smell)."""
    if text_column not in train_df.columns or text_column not in val_df.columns:
        return
    train_texts = set(train_df[text_column].fillna("").astype(str).head(sample))
    val_texts = set(val_df[text_column].fillna("").astype(str).head(sample))
    overlap = len(train_texts & val_texts)
    if overlap:
        logger.warning(
            "Label-leakage check: %d / %d sampled val rows have identical text "
            "in train (text_column=%r). This will inflate validation metrics.",
            overlap, len(val_texts), text_column,
        )
    else:
        logger.info(
            "Label-leakage check: 0 overlap on %d sampled val rows (text_column=%r).",
            len(val_texts), text_column,
        )


def _sanity_batch_test(model, dataloader, device) -> None:
    """Forward one batch and assert per-task losses are sane before training.

    Catches obvious failures (NaN/Inf loss, dead head with loss==0, exploded
    head with loss > 50) BEFORE we burn hours on a broken run.
    """
    try:
        batch = next(iter(dataloader))
    except StopIteration:
        logger.warning("Sanity batch test: dataloader is empty; skipping.")
        return

    if isinstance(batch, dict):
        batch = {
            k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()
        }

    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            outputs = model(**batch)
    finally:
        if was_training:
            model.train()

    task_losses = None
    if isinstance(outputs, dict):
        task_losses = outputs.get("task_losses") or outputs.get("loss_breakdown")
    else:
        task_losses = getattr(outputs, "task_losses", None)

    if not isinstance(task_losses, dict) or not task_losses:
        logger.warning("Sanity batch test: model returned no task_losses; skipping checks.")
        return

    issues = []
    parts = []
    for name, val in task_losses.items():
        if not torch.is_tensor(val):
            continue
        v = float(val.detach().item())
        parts.append(f"{name}={v:.4f}")
        if not math.isfinite(v):
            issues.append(f"{name} is non-finite ({v})")
        elif v == 0.0:
            issues.append(f"{name} is exactly 0.0 (dead head?)")
        elif v > 50.0:
            issues.append(f"{name} is exploded ({v:.2f})")

    logger.info("Sanity batch test: %s", " ".join(parts))
    if issues:
        raise RuntimeError(
            "Sanity batch test failed before training: " + "; ".join(issues)
        )


def _safe_int_series(s, default=0):
    return pd.to_numeric(s, errors="coerce").fillna(default).astype(np.int64)


def _safe_float_series(s, default=0.0):
    return pd.to_numeric(s, errors="coerce").fillna(default).astype(np.float32)


def _entity_series_to_binary(s) -> np.ndarray:
    if s is None:
        return None
    out = s.fillna("").astype(str).str.strip().ne("").astype(np.int64).to_numpy()
    return out


class TruthLensMultiTaskDataset(Dataset):
    """Pre-tokenizes texts and pre-builds label tensors at construction time.

    Removes per-batch tokenization and per-batch Python loops in collate.
    """

    def __init__(self, df, tokenizer, max_length=256, text_column="text"):
        df = df.reset_index(drop=True)
        self.max_length = max_length
        self.text_column = text_column

        # Pre-tokenize the entire split (no padding here — handled by collator)
        texts = df[text_column].fillna("").astype(str).tolist()
        enc = tokenizer(
            texts,
            padding=False,
            truncation=True,
            max_length=max_length,
        )
        self.input_ids = enc["input_ids"]
        self.attention_mask = enc["attention_mask"]
        self.lengths = [len(ids) for ids in self.input_ids]

        # Pre-build label tensors (vectorized via pandas / numpy)
        bias = _safe_int_series(df.get(BIAS_LABEL, 0)).to_numpy()
        ideology = _safe_int_series(df.get(IDEOLOGY_LABEL, 1), default=1).to_numpy()
        propaganda = _safe_int_series(df.get(PROPAGANDA_LABEL, 0)).to_numpy()

        hero_lbl = _safe_int_series(df.get("hero", 0)).to_numpy()
        villain_lbl = _safe_int_series(df.get("villain", 0)).to_numpy()
        victim_lbl = _safe_int_series(df.get("victim", 0)).to_numpy()

        hero_ent = _entity_series_to_binary(df["hero_entities"]) if "hero_entities" in df else np.zeros(len(df), dtype=np.int64)
        villain_ent = _entity_series_to_binary(df["villain_entities"]) if "villain_entities" in df else np.zeros(len(df), dtype=np.int64)
        victim_ent = _entity_series_to_binary(df["victim_entities"]) if "victim_entities" in df else np.zeros(len(df), dtype=np.int64)

        narrative = np.stack(
            [
                np.maximum(hero_lbl, hero_ent),
                np.maximum(villain_lbl, villain_ent),
                np.maximum(victim_lbl, victim_ent),
            ],
            axis=1,
        ).astype(np.float32)

        frame = np.stack(
            [_safe_float_series(df.get(c, 0)).to_numpy() for c in FRAME_COLUMNS],
            axis=1,
        ).astype(np.float32)

        emotion = np.stack(
            [_safe_float_series(df.get(c, 0)).to_numpy() for c in EMOTION_COLUMNS],
            axis=1,
        ).astype(np.float32)

        self.labels_bias = torch.from_numpy(bias).long()
        self.labels_ideology = torch.from_numpy(ideology).long()
        self.labels_propaganda = torch.from_numpy(propaganda).long()
        self.labels_narrative = torch.from_numpy(narrative)
        self.labels_narrative_frame = torch.from_numpy(frame)
        self.labels_emotion = torch.from_numpy(emotion)

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return {
            "input_ids": self.input_ids[idx],
            "attention_mask": self.attention_mask[idx],
            "labels_bias": self.labels_bias[idx],
            "labels_ideology": self.labels_ideology[idx],
            "labels_propaganda": self.labels_propaganda[idx],
            "labels_narrative": self.labels_narrative[idx],
            "labels_narrative_frame": self.labels_narrative_frame[idx],
            "labels_emotion": self.labels_emotion[idx],
        }


# -----------------------------------------------------
# Length-bucketed batch sampler
# -----------------------------------------------------
class BucketSampler(Sampler):
    """Groups examples of similar length into batches to minimize padding."""

    def __init__(self, lengths, batch_size, shuffle=True, bucket_size=100):
        self.batch_size = batch_size
        self.shuffle = shuffle

        indices = list(range(len(lengths)))
        indices.sort(key=lambda i: lengths[i])

        self.buckets = [
            indices[i:i + bucket_size]
            for i in range(0, len(indices), bucket_size)
        ]

    def __iter__(self):
        buckets = [list(b) for b in self.buckets]
        if self.shuffle:
            for b in buckets:
                random.shuffle(b)
            random.shuffle(buckets)

        batch = []
        for bucket in buckets:
            for idx in bucket:
                batch.append(idx)
                if len(batch) == self.batch_size:
                    yield batch
                    batch = []
        if batch:
            yield batch

    def __len__(self):
        total = sum(len(b) for b in self.buckets)
        return math.ceil(total / self.batch_size)

# -----------------------------------------------------
# Load Data
# -----------------------------------------------------

def load_data():

    missing = [p for p in (TRAIN_PATH, VAL_PATH, TEST_PATH) if not p.is_file()]
    if missing:
        searched = "\n  ".join(str(c) for c in _DATA_CANDIDATES)
        raise FileNotFoundError(
            "TruthLens dataset CSV(s) not found:\n  "
            + "\n  ".join(str(p) for p in missing)
            + f"\n\nResolved data dir: {DRIVE_DATA_PATH}"
            + f"\nSearched candidates (first match wins):\n  {searched}"
            + "\n\nFix: either place the CSVs in one of the candidate"
              " directories, or set TRUTHLENS_DATA_DIR=/path/to/your/data"
              " before running."
        )

    # low_memory=False forces a single-pass type inference pass and
    # eliminates the "Columns (...) have mixed types" DtypeWarning that
    # otherwise leaks string/NaN values into label columns. Label columns
    # are subsequently coerced to numeric via _safe_int_series / _safe_float_series.
    # We additionally pin dtypes for known label / text columns so pandas
    # never has to guess on chunk boundaries.
    _explicit_dtypes = {
        TEXT_COLUMN: "string",
        "title": "string",
        BIAS_LABEL: "Int64",
        IDEOLOGY_LABEL: "Int64",
        PROPAGANDA_LABEL: "Int64",
        "hero": "Int64",
        "villain": "Int64",
        "victim": "Int64",
        "hero_entities": "string",
        "villain_entities": "string",
        "victim_entities": "string",
        **{c: "Float64" for c in FRAME_COLUMNS},
        **{c: "Float64" for c in EMOTION_COLUMNS},
    }

    def _read(path):
        # dtype is best-effort: only columns that exist are pinned; pandas
        # ignores unknown dtype keys silently.
        return pd.read_csv(path, low_memory=False, dtype=_explicit_dtypes)

    train_df = _read(TRAIN_PATH)
    val_df = _read(VAL_PATH)
    test_df = _read(TEST_PATH)

    # Label-range sanity probe — catches silently-corrupted label columns
    # (e.g. stray "?", out-of-range categorical IDs). Warn-only: real-world
    # datasets occasionally have ragged rows and we don't want to crash a
    # 6h training run on an off-by-one in row 4 million.
    _label_ranges = {
        BIAS_LABEL: (0, 1),         # binary
        PROPAGANDA_LABEL: (0, 1),    # binary
        IDEOLOGY_LABEL: (0, 4),      # 5-way
    }
    for split_name, df in (("train", train_df), ("val", val_df), ("test", test_df)):
        for col, (lo, hi) in _label_ranges.items():
            if col not in df.columns:
                continue
            s = pd.to_numeric(df[col], errors="coerce")
            n_nan = int(s.isna().sum())
            n_oor = int(((s < lo) | (s > hi)).sum())
            if n_nan or n_oor:
                logger.warning(
                    "Label sanity (%s.%s): %d NaN, %d out-of-range "
                    "(expected [%d,%d]) — coerced via _safe_int_series",
                    split_name, col, n_nan, n_oor, lo, hi,
                )

    for df in (train_df, val_df, test_df):
        if "title" in df.columns and TEXT_COLUMN in df.columns:
            df[TEXT_COLUMN] = df["title"].fillna("").str.cat(
                df[TEXT_COLUMN].fillna(""),
                sep=" ",
            )

    # Empty-text guard (data-contract audit): an all-empty text row
    # tokenizes to padding only and produces a meaningless gradient
    # signal — exactly the "garbled batch" pattern the loss-spike
    # audit flagged. Drop them and log; only fail if a split is
    # entirely empty.
    for split_name, df in (("train", train_df), ("val", val_df), ("test", test_df)):
        if TEXT_COLUMN not in df.columns:
            continue
        empty_mask = df[TEXT_COLUMN].fillna("").astype(str).str.strip().eq("")
        n_empty = int(empty_mask.sum())
        if n_empty:
            logger.warning(
                "Dropping %d empty-text rows from %s (%.2f%%)",
                n_empty, split_name, 100.0 * n_empty / max(1, len(df)),
            )
            df.drop(df.index[empty_mask], inplace=True)
        if len(df) == 0:
            raise RuntimeError(
                f"Split '{split_name}' is empty after dropping empty-text rows; "
                f"check the input CSV's text column ('{TEXT_COLUMN}')."
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

            # The model returns outputs["bias"] = {"logits": ..., "loss": ...}.
            # Earlier code looked under outputs["heads"]["bias"] which never
            # existed — that produced "Test evaluation skipped: no bias logits".
            # Per the output-contract audit: this is a HARD requirement.
            # Missing bias logits is a model contract violation, not a
            # data issue, so we fail fast instead of silently skipping.
            logits = None
            if isinstance(outputs, dict):
                head = outputs.get("bias")
                if isinstance(head, dict):
                    logits = head.get("logits")
                elif torch.is_tensor(head):
                    logits = head
                else:
                    # Backward-compat fallbacks for older output shapes.
                    heads = outputs.get("heads") or outputs.get("logits") or {}
                    if isinstance(heads, dict):
                        logits = heads.get("bias")
                    elif torch.is_tensor(heads):
                        logits = heads
            if logits is None:
                raise RuntimeError(
                    "Model output contract violation: no bias logits in "
                    f"outputs (keys={list(outputs) if isinstance(outputs, dict) else type(outputs)}). "
                    "The bias head must always emit logits — check that "
                    "the checkpoint was loaded with strict=True and that "
                    "bias_head weights are present."
                )

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

        # Cheap label-leakage probe: identical input texts in train+val
        # silently inflate validation metrics. Warn early so the user knows
        # before trusting any val numbers.
        try:
            _check_label_leakage(train_df, val_df, text_column=TEXT_COLUMN)
        except Exception as exc:
            logger.warning("Label-leakage check failed: %s", exc)

        tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)

        # Vectorized collate: pad input_ids/attention_mask to multiple of 8
        # (Tensor Core friendly), then stack pre-built label tensors.
        _LABEL_KEYS = (
            "labels_bias",
            "labels_ideology",
            "labels_propaganda",
            "labels_narrative",
            "labels_narrative_frame",
            "labels_emotion",
        )

        # Dynamic padding via the canonical HF collator. This is the
        # warning-free fast-tokenizer path: transformers detects __call__-style
        # pre-tokenization + DataCollatorWithPadding as the optimal pattern
        # and skips the "use __call__" warning that fires for manual
        # tokenizer.pad() usage.
        _hf_padder = DataCollatorWithPadding(
            tokenizer=tokenizer,
            padding=True,
            pad_to_multiple_of=8,
            return_tensors="pt",
        )

        def collate_fn(batch):
            features = [
                {"input_ids": b["input_ids"], "attention_mask": b["attention_mask"]}
                for b in batch
            ]
            enc = _hf_padder(features)
            labels = {
                "bias": torch.stack([b["labels_bias"] for b in batch]),
                "ideology": torch.stack([b["labels_ideology"] for b in batch]),
                "propaganda": torch.stack([b["labels_propaganda"] for b in batch]),
                "narrative": torch.stack([b["labels_narrative"] for b in batch]),
                "narrative_frame": torch.stack([b["labels_narrative_frame"] for b in batch]),
                "emotion": torch.stack([b["labels_emotion"] for b in batch]),
            }
            return {
                "input_ids": enc["input_ids"],
                "attention_mask": enc["attention_mask"],
                "labels": labels,
            }

        train_dataset = TruthLensMultiTaskDataset(
            train_df,
            tokenizer=tokenizer,
            max_length=max_length,
            text_column=TEXT_COLUMN,
        )

        val_dataset = TruthLensMultiTaskDataset(
            val_df,
            tokenizer=tokenizer,
            max_length=max_length,
            text_column=TEXT_COLUMN,
        )

        test_dataset = TruthLensMultiTaskDataset(
            test_df,
            tokenizer=tokenizer,
            max_length=max_length,
            text_column=TEXT_COLUMN,
        )

        # A100-tuned dataloader throughput
        _num_workers = 4 if _pin else 0
        _persistent = bool(_num_workers)
        _prefetch = 4 if _persistent else None

        # Length-bucketed sampler eliminates padding waste on the train loader.
        train_sampler = BucketSampler(
            train_dataset.lengths,
            batch_size=batch_size,
            shuffle=True,
            bucket_size=max(100, batch_size * 8),
        )

        train_loader = DataLoader(
            train_dataset,
            batch_sampler=train_sampler,
            collate_fn=collate_fn,
            num_workers=_num_workers,
            pin_memory=_pin,
            persistent_workers=_persistent,
            prefetch_factor=_prefetch,
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            collate_fn=collate_fn,
            num_workers=_num_workers,
            pin_memory=_pin,
            persistent_workers=_persistent,
            prefetch_factor=_prefetch,
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

        # Gradient checkpointing trades ~15-25% speed for lower memory.
        # Default OFF (faster); enable via TRUTHLENS_GRADIENT_CHECKPOINTING=1
        # if you hit OOM.
        if os.environ.get("TRUTHLENS_GRADIENT_CHECKPOINTING", "0") == "1":
            if hasattr(model, "encoder") and hasattr(model.encoder, "gradient_checkpointing_enable"):
                model.encoder.gradient_checkpointing_enable()
                logger.info("Gradient checkpointing ENABLED (memory-saving mode)")

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

        # AMP dtype: prefer bf16 on Ampere+ (A100/L4/H100) for native Tensor
        # Core support and numerical stability; fall back to fp16 on T4.
        if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
            _default_amp_dtype = "bf16"
        else:
            _default_amp_dtype = "fp16"

        trainer_config = TrainerConfig(
            epochs=epochs,
            gradient_accumulation_steps=gradient_accumulation_steps,
            device=str(device),
            use_amp=(device.type == "cuda"),
            amp_dtype=os.environ.get("TRUTHLENS_AMP_DTYPE", _default_amp_dtype),
            checkpoint_dir=str(SETTINGS.paths.models_dir / "checkpoints"),
            # Save a step-checkpoint every 4000 optimizer-loop steps in addition
            # to the epoch-end checkpoints handled inside Trainer.train.
            checkpoint_every_steps=int(
                os.environ.get("TRUTHLENS_CHECKPOINT_EVERY_STEPS", "4000")
            ),
            log_every_steps=100,
            validate_every_n_epochs=int(os.environ.get("TRUTHLENS_VALIDATE_EVERY", "2")),
        )

        trainer = Trainer(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            config=trainer_config,
        )

        # --------------------------------------------------
        # Resume from latest checkpoint (Lightning-style explicit)
        # Trainer.__init__ already attempts a resume; this block just
        # surfaces it in the launcher and acts as a fallback if the
        # internal resume was skipped for any reason.
        # --------------------------------------------------
        if (
            trainer.checkpoint_manager is not None
            and trainer.global_step == 0
        ):
            latest_ckpt = trainer.checkpoint_manager.get_latest_checkpoint()
            if latest_ckpt is not None:
                try:
                    logger.info("Resuming from %s", latest_ckpt)
                    trainer.load_checkpoint(str(latest_ckpt), strict=True)
                except Exception as exc:
                    logger.warning("Explicit resume failed (%s); starting fresh", exc)

        # --------------------------------------------------
        # Launcher-level interrupt handler (SIGINT / SIGTERM)
        # Trainer.train() also installs its own scoped handler; this
        # one covers the pre-train setup window and any post-train
        # cleanup so a Ctrl-C never loses progress.
        # --------------------------------------------------
        def _handle_launcher_interrupt(signum, _frame):
            logger.warning(
                "Launcher interrupt %s — saving checkpoint at step %d",
                signum, trainer.global_step,
            )
            try:
                trainer.save_checkpoint(tag="interrupt")
            except Exception as exc:
                logger.error("Launcher checkpoint save failed: %s", exc)
            sys.exit(0)

        for _sig in (signal.SIGINT, signal.SIGTERM):
            try:
                signal.signal(_sig, _handle_launcher_interrupt)
            except (ValueError, OSError):
                pass

        # Sanity batch test — forward one batch and assert per-task losses
        # are finite, non-zero, and not exploded. Aborts fast on a broken
        # data/model wiring before we burn hours of GPU time.
        try:
            _sanity_batch_test(trainer.model, train_loader, device)
        except RuntimeError:
            raise
        except Exception as exc:
            logger.warning("Sanity batch test skipped: %s", exc)

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
