from __future__ import annotations

import logging
import hashlib
from dataclasses import dataclass
from typing import Dict, Optional, List

import pandas as pd

logger = logging.getLogger(__name__)


# =========================================================
# CONFIG
# =========================================================

@dataclass
class LeakageConfig:
    strict: bool = True
    check_near_duplicates: bool = False
    sample_size: int = 10000   # for large datasets
    report_examples: int = 5


# =========================================================
# RESULT
# =========================================================

@dataclass
class LeakageReport:
    train_val_overlap: int = 0
    train_test_overlap: int = 0
    val_test_overlap: int = 0

    examples: Dict[str, List[str]] = None


# =========================================================
# HASH UTILS
# =========================================================

def _hash_text(text: str) -> str:
    return hashlib.md5(text.encode("utf-8")).hexdigest()


def _normalize(text: str) -> str:
    return str(text).strip().lower()


# =========================================================
# CORE CHECK (SINGLE TASK)
# =========================================================

def check_leakage_splits(
    train: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame,
    *,
    config: Optional[LeakageConfig] = None,
) -> LeakageReport:

    config = config or LeakageConfig()

    # normalize + hash
    train_hash = set(train["text"].map(lambda x: _hash_text(_normalize(x))))
    val_hash = set(val["text"].map(lambda x: _hash_text(_normalize(x))))
    test_hash = set(test["text"].map(lambda x: _hash_text(_normalize(x))))

    tv = train_hash & val_hash
    tt = train_hash & test_hash
    vt = val_hash & test_hash

    report = LeakageReport(
        train_val_overlap=len(tv),
        train_test_overlap=len(tt),
        val_test_overlap=len(vt),
        examples={}
    )

    # collect examples
    if config.report_examples > 0:
        report.examples = {
            "train_val": list(tv)[:config.report_examples],
            "train_test": list(tt)[:config.report_examples],
            "val_test": list(vt)[:config.report_examples],
        }

    _handle_result(report, config)

    return report


# =========================================================
# MULTI-TASK CHECK (YOUR CASE)
# =========================================================

def check_leakage_all_tasks(
    datasets: Dict[str, Dict[str, pd.DataFrame]],
    *,
    config: Optional[LeakageConfig] = None,
) -> Dict[str, LeakageReport]:
    """
    datasets = {
        "bias": {"train": df, "val": df, "test": df},
        ...
    }
    """

    results = {}

    for task, splits in datasets.items():

        logger.info("Checking leakage for task: %s", task)

        report = check_leakage_splits(
            splits["train"],
            splits["val"],
            splits["test"],
            config=config,
        )

        results[task] = report

    return results


# =========================================================
# OPTIONAL: NEAR DUPLICATES (ADVANCED)
# =========================================================

def check_near_duplicates(
    df1: pd.DataFrame,
    df2: pd.DataFrame,
    threshold: float = 0.9,
):
    """
    Very basic similarity check (can be replaced with embeddings).
    """

    from difflib import SequenceMatcher

    overlaps = 0

    texts1 = df1["text"].astype(str).tolist()
    texts2 = df2["text"].astype(str).tolist()

    for t1 in texts1:
        for t2 in texts2:
            if SequenceMatcher(None, t1, t2).ratio() > threshold:
                overlaps += 1

    return overlaps


# =========================================================
# HANDLER
# =========================================================

def _handle_result(report: LeakageReport, config: LeakageConfig):

    total = (
        report.train_val_overlap
        + report.train_test_overlap
        + report.val_test_overlap
    )

    if total == 0:
        logger.info("No data leakage detected")
        return

    msg = (
        f"Leakage detected | "
        f"train-val={report.train_val_overlap}, "
        f"train-test={report.train_test_overlap}, "
        f"val-test={report.val_test_overlap}"
    )

    if config.strict:
        raise RuntimeError(msg)
    else:
        logger.warning(msg)