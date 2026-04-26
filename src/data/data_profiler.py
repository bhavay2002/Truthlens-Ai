from __future__ import annotations

import logging
from dataclasses import dataclass, asdict
from typing import Dict, Any, Optional

import pandas as pd

from src.data.class_balance import analyze_task_balance

logger = logging.getLogger(__name__)


# =========================================================
# CONFIG
# =========================================================

@dataclass
class DataProfilerConfig:
    compute_text_stats: bool = True
    compute_duplicates: bool = True
    compute_class_balance: bool = True
    sample_size: Optional[int] = None  # for large datasets


# =========================================================
# RESULT
# =========================================================

@dataclass
class DataProfile:
    rows: int
    columns: int

    # text stats
    avg_text_len: Optional[float] = None
    min_text_len: Optional[int] = None
    max_text_len: Optional[int] = None

    # quality
    duplicate_rows: Optional[int] = None
    empty_text_rows: Optional[int] = None

    # label distribution
    class_balance: Optional[Dict[str, Any]] = None

    # misc
    memory_usage_mb: Optional[float] = None


# =========================================================
# CORE PROFILER
# =========================================================

def profile_dataframe(
    df: pd.DataFrame,
    *,
    task: Optional[str] = None,
    config: Optional[DataProfilerConfig] = None,
) -> DataProfile:

    config = config or DataProfilerConfig()

    # optional sampling (for speed)
    if config.sample_size and len(df) > config.sample_size:
        df = df.sample(config.sample_size, random_state=42)

    profile = DataProfile(
        rows=len(df),
        columns=len(df.columns),
    )

    # -----------------------------------------------------
    # TEXT STATS
    # -----------------------------------------------------
    if config.compute_text_stats and "text" in df.columns:

        lengths = df["text"].astype(str).str.len()

        profile.avg_text_len = float(lengths.mean())
        profile.min_text_len = int(lengths.min())
        profile.max_text_len = int(lengths.max())

        profile.empty_text_rows = int((lengths == 0).sum())

    # -----------------------------------------------------
    # DUPLICATES
    # -----------------------------------------------------
    if config.compute_duplicates and "text" in df.columns:
        profile.duplicate_rows = int(df.duplicated(subset=["text"]).sum())

    # -----------------------------------------------------
    # MEMORY
    # -----------------------------------------------------
    profile.memory_usage_mb = float(df.memory_usage(deep=True).sum()) / (1024 ** 2)

    # -----------------------------------------------------
    # CLASS BALANCE
    # -----------------------------------------------------
    if config.compute_class_balance and task:
        try:
            balance = analyze_task_balance(df, task)
            profile.class_balance = {
                "type": balance.type,
                "distribution": balance.distribution,
                "imbalance": balance.imbalance_detected,
            }
        except Exception as e:
            logger.warning("Class balance failed: %s", e)

    logger.info(
        "Profile | task=%s | rows=%d | avg_len=%.2f",
        task,
        profile.rows,
        profile.avg_text_len or 0.0,
    )

    return profile


# =========================================================
# MULTI-TASK PROFILING (YOUR CASE)
# =========================================================

def profile_all_datasets(
    datasets: Dict[str, Dict[str, pd.DataFrame]],
    *,
    split: str = "train",
    config: Optional[DataProfilerConfig] = None,
) -> Dict[str, DataProfile]:
    """
    datasets = {
        "bias": {"train": df, "val": df, "test": df},
        ...
    }
    """

    results = {}

    for task, splits in datasets.items():

        if split not in splits:
            logger.warning("Missing split %s for task %s", split, task)
            continue

        logger.info("Profiling task: %s", task)

        profile = profile_dataframe(
            splits[split],
            task=task,
            config=config,
        )

        results[task] = profile

    return results


# =========================================================
# EXPORT (OPTIONAL)
# =========================================================

def profile_to_dict(profile: DataProfile) -> Dict[str, Any]:
    return asdict(profile)