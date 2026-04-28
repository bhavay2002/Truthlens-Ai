from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import List, Optional

import pandas as pd

from src.data_processing.data_contracts import CONTRACTS, get_contract

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
    # TEXT CLEANING — vectorized via pandas .str ops (PERF-D1).
    # ~5-10x faster than a Python .map(_clean_text) loop on 100k rows
    # because every step stays in the C-level pandas/regex engine.
    # -----------------------------------------------------
    s = df["text"].astype(str)
    if cfg.strip_html:
        s = s.str.replace(HTML_RE, " ", regex=True)
    if cfg.strip_urls:
        s = s.str.replace(URL_RE, " ", regex=True)
    if cfg.normalize_whitespace:
        s = s.str.replace(WS_RE, " ", regex=True)
    if cfg.lowercase:
        s = s.str.lower()
    df["text"] = s.str.strip()

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
    # DROP DUPLICATES (TEXT-LEVEL) — case-insensitive to stay
    # consistent with leakage_checker._normalize, which lowercases
    # before hashing. Otherwise "Foo" and "foo" both survive dedup
    # but collide in the leakage check, raising a false positive
    # under strict=True. (LEAK-D3)
    # -----------------------------------------------------
    if cfg.drop_duplicates:
        before = len(df)
        norm = df["text"].str.lower()
        df = df.loc[~norm.duplicated()]
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
    Apply task-specific cleaning rules. (CRIT-D1)

    Label columns are pulled from the canonical contracts table
    (``data_contracts.CONTRACTS``) instead of a duplicated lookup —
    so adding/renaming a label in one place keeps cleaning, validation,
    factory, and sampler perfectly in sync.
    """

    cfg = config or DataCleaningConfig()

    if task in CONTRACTS:
        label_cols: List[str] = list(get_contract(task).label_columns)
    else:
        # Unknown task → behave as before (no label-aware cleaning),
        # but warn loudly so a typo doesn't silently disable label fill.
        logger.warning("clean_for_task called with unknown task=%s — skipping label-aware cleaning", task)
        label_cols = []

    return clean_dataframe(
        df,
        config=cfg,
        label_cols=label_cols,
    )