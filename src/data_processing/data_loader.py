from __future__ import annotations

import logging
import hashlib
from pathlib import Path
from typing import Optional, List, Dict, Any

import pandas as pd

logger = logging.getLogger(__name__)


# =========================================================
# CONFIG
# =========================================================

DEFAULT_ENCODING = "utf-8"
FALLBACK_ENCODING = "latin-1"


# =========================================================
# FILE HASHING (INTEGRITY)
# =========================================================

def compute_md5(path: Path) -> str:
    """
    Compute MD5 hash for file integrity checks.
    """
    hash_md5 = hashlib.md5()

    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            hash_md5.update(chunk)

    return hash_md5.hexdigest()


# =========================================================
# CORE LOADERS
# =========================================================

def load_csv(
    path: Path,
    *,
    usecols: Optional[List[str]] = None,
    dtype: Optional[Dict[str, Any]] = None,
    encoding: str = DEFAULT_ENCODING,
) -> pd.DataFrame:
    """
    Robust CSV loader with encoding fallback.
    """

    try:
        df = pd.read_csv(
            path,
            encoding=encoding,
            usecols=usecols,
            dtype=dtype,
        )
    except UnicodeDecodeError:
        logger.warning("Encoding fallback for %s", path)
        df = pd.read_csv(
            path,
            encoding=FALLBACK_ENCODING,
            usecols=usecols,
            dtype=dtype,
        )

    return df


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
    """
    Unified loader for CSV/JSON/Parquet.

    Returns:
        pd.DataFrame
    """

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

    # -------------------------
    # HASH (optional)
    # -------------------------
    if compute_hash:
        file_hash = compute_md5(path)
        logger.info("MD5(%s) = %s", path.name, file_hash)

    logger.info("Loaded %d rows, %d columns", len(df), len(df.columns))

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
    """
    Generator for large CSV files.
    """

    try:
        reader = pd.read_csv(
            path,
            encoding=DEFAULT_ENCODING,
            chunksize=chunksize,
            usecols=usecols,
            dtype=dtype,
        )
    except UnicodeDecodeError:
        logger.warning("Encoding fallback for chunked read: %s", path)
        reader = pd.read_csv(
            path,
            encoding=FALLBACK_ENCODING,
            chunksize=chunksize,
            usecols=usecols,
            dtype=dtype,
        )

    for chunk in reader:
        yield chunk


# =========================================================
# VALIDATION HOOK (OPTIONAL)
# =========================================================

def enforce_required_columns(
    df: pd.DataFrame,
    required_cols: List[str],
):
    missing = [c for c in required_cols if c not in df.columns]

    if missing:
        raise ValueError(f"Missing required columns: {missing}")