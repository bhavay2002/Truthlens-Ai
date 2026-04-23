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
    configure_cuda_kernels,
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
_VAL_FILE = "unified_dataset_validation.csv"
_TEST_FILE = "unified_dataset_test.csv"


def _has_all_dataset_splits(path: Path) -> bool:
    return all((path / name).is_file() for name in (_TRAIN_FILE, _VAL_FILE, _TEST_FILE))


def _search_dataset_dir(root: Path, max_depth: int = 3) -> Path | None:
    """Find the nearest directory containing all unified dataset splits."""
    try:
        root = root.expanduser()
    except Exception:
        return None

    if _has_all_dataset_splits(root):
        return root
    if not root.exists() or not root.is_dir() or max_depth <= 0:
        return None

    try:
        for child in root.iterdir():
            if not child.is_dir():
                continue
            found = _search_dataset_dir(child, max_depth=max_depth - 1)
            if found is not None:
                return found
    except OSError:
        return None
    return None


def _resolve_data_dir() -> Path:
    env = os.environ.get("TRUTHLENS_DATA_DIR")
    if env:
        env_path = Path(env).expanduser()
        found = _search_dataset_dir(env_path, max_depth=4)
        return found or env_path
    for cand in _DATA_CANDIDATES:
        found = _search_dataset_dir(cand, max_depth=3)
        if found is not None:
            return found
    for root in (
        Path("/teamspace/studios/this_studio"),
        Path(__file__).resolve().parent,
        Path("/content"),
    ):
        found = _search_dataset_dir(root, max_depth=4)
        if found is not None:
            return found
    # Nothing found — return the first repo-local candidate so the
    # FileNotFoundError below is actionable (path printed in the error).
    return Path(__file__).resolve().parent / "data"


DRIVE_DATA_PATH = _resolve_data_dir()

TRAIN_PATH = DRIVE_DATA_PATH / _TRAIN_FILE
VAL_PATH = DRIVE_DATA_PATH / _VAL_FILE
TEST_PATH = DRIVE_DATA_PATH / _TEST_FILE

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
    """Detect train/val overlap; fail fast when threshold is exceeded."""
    if text_column not in train_df.columns or text_column not in val_df.columns:
        return
    train_texts = set(train_df[text_column].fillna("").astype(str).head(sample))
    val_texts = set(val_df[text_column].fillna("").astype(str).head(sample))
    overlap = len(train_texts & val_texts)
    train_hashes = {
        hashlib.md5(text.encode("utf-8")).hexdigest()
        for text in train_df[text_column].fillna("").astype(str).head(sample)
    }
    val_hashes = {
        hashlib.md5(text.encode("utf-8")).hexdigest()
        for text in val_df[text_column].fillna("").astype(str).head(sample)
    }
    hash_overlap = len(train_hashes & val_hashes)

    if overlap == 0 and hash_overlap == 0:
        logger.info(
            "Label-leakage check: 0 overlap on %d sampled val rows (text_column=%r).",
            len(val_texts), text_column,
        )
        return

    msg = (
        "Label-leakage check detected sampled overlap "
        "(raw=%d, hash=%d, val_sample=%d, text_column=%r)."
    )
    max_overlap = int(os.environ.get("TRUTHLENS_MAX_TEXT_OVERLAP", "0"))
    if max(overlap, hash_overlap) > max_overlap:
        raise RuntimeError(
            (msg + " Allowed threshold=%d.")
            % (overlap, hash_overlap, len(val_texts), text_column, max_overlap)
        )
    logger.warning(msg, overlap, hash_overlap, len(val_texts), text_column)


def _log_dataset_label_distribution(df: pd.DataFrame, split_name: str) -> None:
    """Log label distributions so class imbalance is visible before training."""
    label_columns = [BIAS_LABEL, IDEOLOGY_LABEL, PROPAGANDA_LABEL]
    total = max(1, len(df))
    for col in label_columns:
        if col not in df.columns:
            continue
        counts = df[col].value_counts(dropna=False)
        parts = [f"{k}:{v}({(v / total) * 100:.2f}%)" for k, v in counts.items()]
        logger.info(
            "Label distribution | split=%s | column=%s | %s",
            split_name,
            col,
            " ".join(parts),
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


def _validate_model_output_contract(model, tokenizer, device) -> None:
    """Fail fast if required task outputs are missing after model init/load."""
    was_training = model.training
    model.eval()
    try:
        dummy = tokenizer(
            ["contract validation sample"],
            return_tensors="pt",
            truncation=True,
            max_length=32,
        )
        dummy = {k: v.to(device) for k, v in dummy.items()}
        with torch.no_grad():
            outputs = model(**dummy)
    finally:
        if was_training:
            model.train()

    if not isinstance(outputs, dict):
        raise RuntimeError(
            "Model output contract violation: forward() must return a dict."
        )

    required_tasks = ("bias", "ideology", "propaganda", "narrative", "narrative_frame", "emotion")
    for task_name in required_tasks:
        head = outputs.get(task_name)
        if not isinstance(head, dict):
            raise RuntimeError(
                f"Model output contract violation: missing task output '{task_name}'. "
                f"Available keys: {list(outputs.keys())}"
            )
        logits = head.get("logits")
        if not torch.is_tensor(logits):
            raise RuntimeError(
                f"Model output contract violation: task '{task_name}' missing tensor logits."
            )
        if logits.shape[-1] <= 0:
            raise RuntimeError(
                f"Model output contract violation: task '{task_name}' has invalid logits shape {tuple(logits.shape)}."
            )


def _strict_int_series(
    s: pd.Series,
    *,
    column_name: str,
    split_name: str = "dataset",
    allowed_values: set[int] | None = None,
    allow_na: bool = False,
) -> pd.Series:
    """Parse integer labels with fail-fast validation (no silent coercion)."""
    normalized = s.astype("string").str.strip()
    parsed = pd.to_numeric(normalized, errors="raise")
    if not allow_na and parsed.isna().any():
        raise RuntimeError(
            f"{split_name}.{column_name} contains NaN values after parsing."
        )
    parsed_int = parsed.astype("Int64")
    if allowed_values is not None:
        invalid = ~parsed_int.isin(sorted(allowed_values))
        if allow_na:
            invalid &= parsed_int.notna()
        if invalid.any():
            examples = (
                normalized[invalid]
                .dropna()
                .astype(str)
                .head(5)
                .tolist()
            )
            raise RuntimeError(
                f"{split_name}.{column_name} has invalid labels. "
                f"Allowed={sorted(allowed_values)} Examples={examples}"
            )
    return parsed_int


def _strict_float_series(
    s: pd.Series,
    *,
    column_name: str,
    split_name: str = "dataset",
    allow_na: bool = False,
) -> pd.Series:
    """Parse float features with fail-fast validation."""
    normalized = s.astype("string").str.strip()
    parsed = pd.to_numeric(normalized, errors="raise")
    if not allow_na and parsed.isna().any():
        raise RuntimeError(
            f"{split_name}.{column_name} contains NaN values after parsing."
        )
    return parsed.astype("Float32")


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

        # Pre-build label tensors from pre-validated columns.
        missing_label_value = -100
        missing_multilabel_value = float(missing_label_value)

        bias = (
            df[BIAS_LABEL]
            .fillna(missing_label_value)
            .astype(np.int64)
            .to_numpy()
        )
        ideology = (
            df[IDEOLOGY_LABEL]
            .fillna(missing_label_value)
            .astype(np.int64)
            .to_numpy()
        )
        propaganda = (
            df[PROPAGANDA_LABEL]
            .fillna(missing_label_value)
            .astype(np.int64)
            .to_numpy()
        )

        hero_raw = df["hero"]
        villain_raw = df["villain"]
        victim_raw = df["victim"]
        hero_lbl = hero_raw.fillna(0).astype(np.int64).to_numpy()
        villain_lbl = villain_raw.fillna(0).astype(np.int64).to_numpy()
        victim_lbl = victim_raw.fillna(0).astype(np.int64).to_numpy()

        hero_ent = _entity_series_to_binary(df["hero_entities"]) if "hero_entities" in df else np.zeros(len(df), dtype=np.int64)
        villain_ent = _entity_series_to_binary(df["villain_entities"]) if "villain_entities" in df else np.zeros(len(df), dtype=np.int64)
        victim_ent = _entity_series_to_binary(df["victim_entities"]) if "victim_entities" in df else np.zeros(len(df), dtype=np.int64)

        narrative_valid = (
            hero_raw.notna()
            | villain_raw.notna()
            | victim_raw.notna()
            | (hero_ent > 0)
            | (villain_ent > 0)
            | (victim_ent > 0)
        ).to_numpy()

        narrative = np.stack(
            [
                np.maximum(hero_lbl, hero_ent),
                np.maximum(villain_lbl, villain_ent),
                np.maximum(victim_lbl, victim_ent),
            ],
            axis=1,
        ).astype(np.float32)
        narrative[~narrative_valid] = missing_multilabel_value

        frame = np.stack(
            [df[c].fillna(missing_multilabel_value).astype(np.float32).to_numpy() for c in FRAME_COLUMNS],
            axis=1,
        ).astype(np.float32)
        frame_valid = df[FRAME_COLUMNS].notna().any(axis=1).to_numpy()
        frame[~frame_valid] = missing_multilabel_value

        emotion = np.stack(
            [df[c].fillna(missing_multilabel_value).astype(np.float32).to_numpy() for c in EMOTION_COLUMNS],
            axis=1,
        ).astype(np.float32)
        emotion_valid = df[EMOTION_COLUMNS].notna().any(axis=1).to_numpy()
        emotion[~emotion_valid] = missing_multilabel_value

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

    # Data contract: pin dtypes at read time so pandas never infers mixed types.
    # low_memory=False prevents chunk-wise inference drift.
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
        # dtype is best-effort: pandas ignores unknown dtype keys.
        return pd.read_csv(
            path,
            low_memory=False,
            dtype=_explicit_dtypes,
            keep_default_na=True,
            na_values=["", "NA", "N/A", "null", "None"],
        )

    train_df = _read(TRAIN_PATH)
    val_df = _read(VAL_PATH)
    test_df = _read(TEST_PATH)

    # Hard schema + value validation. Any violation stops the run.
    _required_columns = [
        TEXT_COLUMN,
        BIAS_LABEL,
        IDEOLOGY_LABEL,
        PROPAGANDA_LABEL,
        "hero",
        "villain",
        "victim",
        *FRAME_COLUMNS,
        *EMOTION_COLUMNS,
    ]
    _label_allowed = {
        BIAS_LABEL: {0, 1},
        PROPAGANDA_LABEL: {0, 1},
        IDEOLOGY_LABEL: {0, 1, 2, 3, 4},
        "hero": {0, 1},
        "villain": {0, 1},
        "victim": {0, 1},
    }

    for split_name, df in (("train", train_df), ("val", val_df), ("test", test_df)):
        missing_cols = [c for c in _required_columns if c not in df.columns]
        if missing_cols:
            raise RuntimeError(
                f"{split_name} split missing required columns: {missing_cols}"
            )

        for col, allowed_values in _label_allowed.items():
            df[col] = _strict_int_series(
                df[col],
                column_name=col,
                split_name=split_name,
                allowed_values=allowed_values,
                allow_na=True,
            )

        for col in [*FRAME_COLUMNS, *EMOTION_COLUMNS]:
            df[col] = _strict_float_series(
                df[col],
                column_name=col,
                split_name=split_name,
                allow_na=True,
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

    # -------------------------------------------------------------
    # Phase-2 distribution audit (multi-task playbook). Logs per-task
    # positive / negative / NA coverage so silent label-collapse and
    # extreme imbalance surface BEFORE training starts. Failure modes
    # this catches:
    #   * a task is 99% NA → the head will look "starved" in trainer logs
    #   * a multi-class task is 99% one class → AUC looks fine but the
    #     head learned the prior, not the signal
    #   * a multi-label task has zero positives for some labels → the
    #     corresponding sigmoid output saturates at 0
    # -------------------------------------------------------------
    _MC_TASKS = (
        ("bias", BIAS_LABEL, {0, 1}),
        ("ideology", IDEOLOGY_LABEL, {0, 1, 2}),
        ("propaganda", PROPAGANDA_LABEL, {0, 1}),
    )
    _ML_TASKS = (
        ("narrative_frame", FRAME_COLUMNS),
        ("emotion", EMOTION_COLUMNS),
    )
    for split_name, _df in (("train", train_df), ("val", val_df), ("test", test_df)):
        n = max(1, len(_df))
        for task_name, col, allowed in _MC_TASKS:
            if col not in _df.columns:
                continue
            s = _df[col]
            na = int(s.isna().sum())
            counts = s.dropna().astype("Int64").value_counts().to_dict()
            class_str = " ".join(
                f"c{int(k)}={int(v)}({100.0 * int(v) / n:.1f}%)"
                for k, v in sorted(counts.items())
            )
            logger.info(
                "[label-audit] %s/%s na=%d(%.1f%%) %s",
                split_name, task_name, na, 100.0 * na / n, class_str,
            )
            if counts:
                top_frac = max(counts.values()) / max(1, sum(counts.values()))
                if top_frac > 0.95:
                    logger.warning(
                        "[label-audit] %s/%s is %.1f%% one class — head will likely "
                        "learn the prior. Consider class weights or oversampling.",
                        split_name, task_name, 100.0 * top_frac,
                    )
        for task_name, cols in _ML_TASKS:
            present = [c for c in cols if c in _df.columns]
            if not present:
                continue
            sub = _df[present]
            row_na = int(sub.isna().all(axis=1).sum())
            pos_per_label = (sub.fillna(0).astype(float) > 0.5).sum(axis=0)
            pos_str = " ".join(
                f"{c}={int(pos_per_label[c])}({100.0 * int(pos_per_label[c]) / n:.1f}%)"
                for c in present
            )
            logger.info(
                "[label-audit] %s/%s row_all_na=%d(%.1f%%) pos: %s",
                split_name, task_name, row_na, 100.0 * row_na / n, pos_str,
            )
            zero_pos = [c for c in present if int(pos_per_label[c]) == 0]
            if zero_pos:
                logger.warning(
                    "[label-audit] %s/%s has labels with ZERO positives in this split: %s",
                    split_name, task_name, zero_pos,
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

            bias_labels = labels["bias"]
            valid_mask = bias_labels.ne(-100)
            if not bool(valid_mask.any()):
                continue

            y_true.extend(bias_labels[valid_mask].cpu().tolist())
            y_pred.extend(preds[valid_mask].cpu().tolist())
            selected_probs = probs[valid_mask]
            y_proba.extend(
                selected_probs[:, 1].cpu().tolist()
                if selected_probs.shape[-1] > 1
                else selected_probs.squeeze(-1).cpu().tolist()
            )

    if not y_true:
        raise RuntimeError(
            "Test evaluation contract violation: no bias labels/predictions were collected."
        )

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

        # CUDA diagnostic block — printed up-front so a CPU fallback is
        # impossible to miss in the logs.
        _cuda_avail = torch.cuda.is_available()
        logger.info("Training device: %s", device)
        logger.info(
            "CUDA available=%s | torch.version.cuda=%s | device_count=%d",
            _cuda_avail,
            getattr(torch.version, "cuda", None),
            torch.cuda.device_count() if _cuda_avail else 0,
        )
        if _cuda_avail:
            logger.info("CUDA device name: %s", torch.cuda.get_device_name(0))

        # Hard GPU gate. Set TRUTHLENS_REQUIRE_GPU=1 (default on Lightning AI
        # Studios / Colab GPU runtimes) to fail loud instead of silently
        # training on CPU at 1/100th throughput. Opt-out with =0 for local
        # smoke tests on a laptop.
        _require_gpu = os.environ.get("TRUTHLENS_REQUIRE_GPU", "0").strip().lower() in {
            "1", "true", "yes", "on",
        }
        if _require_gpu and not _cuda_avail:
            raise RuntimeError(
                "TRUTHLENS_REQUIRE_GPU=1 but CUDA is not available. "
                "Refusing to train on CPU. Either run on a GPU runtime "
                "(nvidia-smi must work, torch must be a CUDA build) or "
                "unset TRUTHLENS_REQUIRE_GPU for an explicit CPU run."
            )

        # C8: cudnn.benchmark only meaningful on CUDA + cuDNN
        if _cuda_avail and torch.backends.cudnn.is_available():
            torch.backends.cudnn.benchmark = True
            configure_cuda_kernels()

        # M1: gate pin_memory on CUDA availability
        _pin = torch.cuda.is_available()

        # --------------------------------------------------
        # Data
        # --------------------------------------------------

        train_df, val_df, test_df = load_data()
        _log_dataset_label_distribution(train_df, "train")
        _log_dataset_label_distribution(val_df, "val")
        _log_dataset_label_distribution(test_df, "test")

        # Defensive dedup BEFORE the leak check. Even if upstream split
        # CSVs contain duplicate text rows or cross-split overlaps, we
        # remove them here so leakage is structurally impossible. Train
        # is the loser in any train/val or train/test collision (we keep
        # the eval split intact so reported metrics stay comparable).
        for split_name, df in (("train", train_df), ("val", val_df), ("test", test_df)):
            before = len(df)
            df.drop_duplicates(subset=[TEXT_COLUMN], keep="first", inplace=True)
            removed = before - len(df)
            if removed:
                logger.warning(
                    "Intra-split dedup: removed %d duplicate-text rows from %s "
                    "(%.2f%%)",
                    removed, split_name, 100.0 * removed / max(1, before),
                )

        _val_texts = set(val_df[TEXT_COLUMN].fillna("").astype(str))
        _test_texts = set(test_df[TEXT_COLUMN].fillna("").astype(str))
        _eval_texts = _val_texts | _test_texts
        _train_text_col = train_df[TEXT_COLUMN].fillna("").astype(str)
        _overlap_mask = _train_text_col.isin(_eval_texts)
        _overlap_count = int(_overlap_mask.sum())
        if _overlap_count:
            logger.warning(
                "Cross-split leakage: dropping %d train rows that also appear "
                "in val/test (%.2f%% of train).",
                _overlap_count,
                100.0 * _overlap_count / max(1, len(train_df)),
            )
            train_df.drop(train_df.index[_overlap_mask.to_numpy()], inplace=True)
            train_df.reset_index(drop=True, inplace=True)

        logger.info(
            "Post-dedup sizes — train: %d  val: %d  test: %d",
            len(train_df), len(val_df), len(test_df),
        )

        # Final leak check is FATAL. After dedup it must pass; if it
        # doesn't, something is structurally wrong (e.g. text column
        # mismatch) and we refuse to train on contaminated data.
        _check_label_leakage(train_df, val_df, text_column=TEXT_COLUMN)
        _check_label_leakage(train_df, test_df, text_column=TEXT_COLUMN)

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

        # ---- Phase-2 oversampling (multi-task playbook). Opt-in via
        # TRUTHLENS_OVERSAMPLE=1. Replaces the bucketed sampler with a
        # WeightedRandomSampler whose per-sample weight is the inverse
        # frequency of its bias class, NA-rows down-weighted. Gives rare
        # classes proportionally more exposure; trades padding-efficiency
        # for label balance.
        _use_weighted_sampler = (
            os.environ.get("TRUTHLENS_OVERSAMPLE", "1") == "1"
        )

        if _use_weighted_sampler:
            from torch.utils.data import WeightedRandomSampler
            _bias_arr = train_df[BIAS_LABEL].fillna(-100).astype("int64").to_numpy()
            _valid_mask = _bias_arr != -100
            _weights = np.ones(len(_bias_arr), dtype=np.float64)
            if _valid_mask.any():
                _classes, _counts = np.unique(_bias_arr[_valid_mask], return_counts=True)
                _inv = {int(c): 1.0 / float(n) for c, n in zip(_classes, _counts)}
                for _i, _v in enumerate(_bias_arr):
                    if _v == -100:
                        # NA-bias rows still carry other-task labels —
                        # keep them in the rotation but at the median
                        # inverse-frequency weight to avoid starvation.
                        _weights[_i] = float(np.median(list(_inv.values())))
                    else:
                        _weights[_i] = _inv[int(_v)]
            train_sampler = WeightedRandomSampler(
                weights=torch.as_tensor(_weights, dtype=torch.double),
                num_samples=len(train_dataset),
                replacement=True,
            )
            logger.info(
                "WeightedRandomSampler ENABLED — class-balanced oversampling "
                "(bias-classes=%s)",
                {int(c): int(n) for c, n in zip(*np.unique(_bias_arr[_valid_mask], return_counts=True))}
                if _valid_mask.any() else {},
            )
            train_loader = DataLoader(
                train_dataset,
                sampler=train_sampler,
                batch_size=batch_size,
                collate_fn=collate_fn,
                num_workers=_num_workers,
                pin_memory=_pin,
                persistent_workers=_persistent,
                prefetch_factor=_prefetch,
                drop_last=True,
            )
        else:
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
        _validate_model_output_contract(model, tokenizer, device)

        # Gradient checkpointing trades ~15-25% speed for lower memory.
        # Default OFF (faster); enable via TRUTHLENS_GRADIENT_CHECKPOINTING=1
        # if you hit OOM.
        if os.environ.get("TRUTHLENS_GRADIENT_CHECKPOINTING", "0") == "1":
            if hasattr(model, "encoder") and hasattr(model.encoder, "gradient_checkpointing_enable"):
                model.encoder.gradient_checkpointing_enable()
                logger.info("Gradient checkpointing ENABLED (memory-saving mode)")

        if hasattr(model, "config") and hasattr(model.config, "use_flash_attention"):
            model.config.use_flash_attention = True

        # ---- Phase-3 EMA-coverage task weighting (multi-task playbook).
        # Opt-in via TRUTHLENS_EMA_TASK_WEIGHTING=1. Boosts under-supervised
        # heads' gradient contribution proportionally to how often they are
        # masked out, without re-tuning the static per-task weights.
        if os.environ.get("TRUTHLENS_EMA_TASK_WEIGHTING", "1") == "1":
            if hasattr(model, "multitask_loss") and hasattr(
                model.multitask_loss, "enable_ema_weighting"
            ):
                _alpha = float(os.environ.get("TRUTHLENS_EMA_ALPHA", "0.1"))
                _floor = float(os.environ.get("TRUTHLENS_EMA_FLOOR", "0.05"))
                _cap = float(os.environ.get("TRUTHLENS_EMA_CAP", "10.0"))
                model.multitask_loss.enable_ema_weighting(
                    alpha=_alpha, floor=_floor, cap=_cap,
                )
                logger.info(
                    "EMA-coverage task weighting ENABLED "
                    "(alpha=%.3f floor=%.3f cap=%.2f)",
                    _alpha, _floor, _cap,
                )

        # ---- Phase-4 Kendall-uncertainty TaskBalancer (multi-task playbook).
        # Opt-in via TRUTHLENS_TASK_BALANCER=1. Replaces the naive weighted
        # sum across heads with a learnable log-variance combination so the
        # model finds its own per-task scaling. Attached BEFORE optimizer
        # construction below so balancer params get picked up.
        if os.environ.get("TRUTHLENS_TASK_BALANCER", "1") == "1":
            try:
                from src.training.instrumentation import TaskBalancer
                if hasattr(model, "multitask_loss") and hasattr(
                    model.multitask_loss, "attach_task_balancer"
                ):
                    _tasks = list(model.multitask_loss.task_configs.keys())
                    _balancer = TaskBalancer(_tasks).to(device)
                    model.multitask_loss.attach_task_balancer(_balancer)
                    logger.info(
                        "Kendall TaskBalancer ATTACHED for tasks=%s", _tasks,
                    )
            except Exception as _exc:
                logger.warning(
                    "TRUTHLENS_TASK_BALANCER=1 but attach failed: %s", _exc,
                )

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
        _validate_model_output_contract(trainer.model, tokenizer, device)

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

        _evaluate_on_test(trainer.model, test_loader, device)

        logger.info("Pipeline finished | history=%s", history)

    except Exception as e:

        logger.error("Training failed: %s", e, exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
