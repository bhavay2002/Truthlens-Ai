"""
Unified file loader for CSV / JSON(L) / Parquet.

Encoding fallback is logged loudly so silent corruption does not slip
through.
"""

from __future__ import annotations

import logging
import hashlib
from pathlib import Path
from typing import Optional, List, Dict, Any

import pandas as pd

logger = logging.getLogger(__name__)

DEFAULT_ENCODING = "utf-8"
FALLBACK_ENCODING = "latin-1"


# =========================================================
# FILE HASHING (INTEGRITY)
# =========================================================

def compute_md5(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# =========================================================
# CORE LOADERS
# =========================================================

def load_csv(
    path: Path,
    *,
    usecols: Optional[List[str]] = None,
    dtype: Optional[Dict[str, Any]] = None,
    encoding: str = DEFAULT_ENCODING,
    na_values: Optional[List[str]] = None,
) -> pd.DataFrame:
    common = dict(usecols=usecols, dtype=dtype, na_values=na_values, low_memory=False)
    try:
        return pd.read_csv(path, encoding=encoding, **common)
    except UnicodeDecodeError:
        logger.warning(
            "Encoding fallback %s → %s for %s",
            encoding, FALLBACK_ENCODING, path,
        )
        return pd.read_csv(path, encoding=FALLBACK_ENCODING, **common)


def load_json(path: Path) -> pd.DataFrame:
    return pd.read_json(path, lines=True)


def load_parquet(path: Path) -> pd.DataFrame:
    return pd.read_parquet(path)


# =========================================================
# GENERIC LOADER
# =========================================================

def load_dataframe(
    path: Path,
    *,
    usecols: Optional[List[str]] = None,
    dtype: Optional[Dict[str, Any]] = None,
    compute_hash: bool = False,
) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    suffix = path.suffix.lower()
    logger.info("Loading dataset: %s", path)

    if suffix == ".csv":
        df = load_csv(path, usecols=usecols, dtype=dtype)
    elif suffix in (".json", ".jsonl"):
        df = load_json(path)
    elif suffix == ".parquet":
        df = load_parquet(path)
    else:
        raise ValueError(f"Unsupported file format: {suffix}")

    if compute_hash:
        logger.info("MD5(%s) = %s", path.name, compute_md5(path))

    logger.info("Loaded %d rows × %d cols", len(df), len(df.columns))
    return df


# =========================================================
# CHUNKED LOADER (LARGE DATA)
# =========================================================

def load_csv_in_chunks(
    path: Path,
    *,
    chunksize: int = 100_000,
    usecols: Optional[List[str]] = None,
    dtype: Optional[Dict[str, Any]] = None,
):
    common = dict(chunksize=chunksize, usecols=usecols, dtype=dtype, low_memory=False)
    try:
        reader = pd.read_csv(path, encoding=DEFAULT_ENCODING, **common)
    except UnicodeDecodeError:
        logger.warning("Encoding fallback for chunked read: %s", path)
        reader = pd.read_csv(path, encoding=FALLBACK_ENCODING, **common)
    yield from reader


# =========================================================
# COLUMN GUARD
# =========================================================

def enforce_required_columns(df: pd.DataFrame, required_cols: List[str]) -> None:
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
