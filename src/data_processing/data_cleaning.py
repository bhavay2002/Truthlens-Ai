from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import List, Optional, Dict, Any

import pandas as pd

logger = logging.getLogger(__name__)


# =========================================================
# CONFIG
# =========================================================

@dataclass
class DataCleaningConfig:
    drop_duplicates: bool = True
    drop_empty_text: bool = True
    normalize_whitespace: bool = True
    lowercase: bool = False              # keep False for most NLP tasks
    strip_urls: bool = False             # optional (can remove signal)
    strip_html: bool = True
    min_text_len: int = 3
    max_text_len: int = 20000

    # label handling
    fill_missing_labels: bool = False    # usually False (let validator fail)
    label_fill_value: int = 0

    # reporting
    log_stats: bool = True


# =========================================================
# REGEX
# =========================================================

URL_RE = re.compile(r"https?://\S+|www\.\S+")
HTML_RE = re.compile(r"<.*?>")
WS_RE = re.compile(r"\s+")


# =========================================================
# TEXT NORMALIZATION
# =========================================================

def _clean_text(text: str, cfg: DataCleaningConfig) -> str:
    if not isinstance(text, str):
        return ""

    t = text

    if cfg.strip_html:
        t = HTML_RE.sub(" ", t)

    if cfg.strip_urls:
        t = URL_RE.sub(" ", t)

    if cfg.normalize_whitespace:
        t = WS_RE.sub(" ", t)

    if cfg.lowercase:
        t = t.lower()

    return t.strip()


# =========================================================
# CORE CLEANING
# =========================================================

def clean_dataframe(
    df: pd.DataFrame,
    *,
    config: Optional[DataCleaningConfig] = None,
    label_cols: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Clean a dataframe in a controlled, reproducible way.

    Args:
        df: input dataframe (must contain 'text')
        config: cleaning behavior
        label_cols: optional label columns to process

    Returns:
        cleaned dataframe
    """

    if "text" not in df.columns:
        raise ValueError("Missing 'text' column")

    cfg = config or DataCleaningConfig()

    orig_rows = len(df)

    # -----------------------------------------------------
    # COPY (avoid side-effects)
    # -----------------------------------------------------
    df = df.copy()

    # -----------------------------------------------------
    # TEXT CLEANING
    # -----------------------------------------------------
    df["text"] = df["text"].astype(str).map(lambda x: _clean_text(x, cfg))

    # -----------------------------------------------------
    # DROP EMPTY / SHORT TEXT
    # -----------------------------------------------------
    if cfg.drop_empty_text:
        mask = (
            df["text"].notna()
            & (df["text"].str.len() >= cfg.min_text_len)
            & (df["text"].str.len() <= cfg.max_text_len)
        )
        df = df[mask]

    # -----------------------------------------------------
    # DROP DUPLICATES (TEXT-LEVEL)
    # -----------------------------------------------------
    if cfg.drop_duplicates:
        before = len(df)
        df = df.drop_duplicates(subset=["text"])
        removed = before - len(df)
        if cfg.log_stats and removed > 0:
            logger.info("Removed %d duplicate rows", removed)

    # -----------------------------------------------------
    # LABEL HANDLING (OPTIONAL)
    # -----------------------------------------------------
    if label_cols and cfg.fill_missing_labels:
        for col in label_cols:
            if col in df.columns:
                df[col] = df[col].fillna(cfg.label_fill_value)

    # -----------------------------------------------------
    # FINAL RESET
    # -----------------------------------------------------
    df = df.reset_index(drop=True)

    if cfg.log_stats:
        logger.info(
            "Data cleaned | rows: %d → %d",
            orig_rows,
            len(df),
        )

    return df


# =========================================================
# TASK-AWARE CLEANING (OPTIONAL)
# =========================================================

def clean_for_task(
    df: pd.DataFrame,
    task: str,
    *,
    config: Optional[DataCleaningConfig] = None,
) -> pd.DataFrame:
    """
    Apply task-specific cleaning rules.
    """

    cfg = config or DataCleaningConfig()

    TASK_LABELS: Dict[str, List[str]] = {
        "bias": ["bias_label"],
        "ideology": ["ideology_label"],
        "propaganda": ["propaganda_label"],
        "frame": ["CO", "EC", "HI", "MO", "RE"],
        "narrative": ["hero", "villain", "victim"],
        "emotion": [f"emotion_{i}" for i in range(20)],
    }

    label_cols = TASK_LABELS.get(task, [])

    return clean_dataframe(
        df,
        config=cfg,
        label_cols=label_cols,
    )