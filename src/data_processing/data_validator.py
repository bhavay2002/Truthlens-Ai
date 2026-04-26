from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional

import pandas as pd

logger = logging.getLogger(__name__)


# =========================================================
# CONFIG
# =========================================================

@dataclass(frozen=True)
class DataValidatorConfig:
    strict: bool = True                 # raise on errors
    check_text: bool = True
    min_text_len: int = 3
    max_text_len: int = 10000

    # label checks
    enforce_label_range: bool = True
    enforce_binary_multilabel: bool = True

    # reporting
    sample_errors: int = 5


# =========================================================
# RESULT
# =========================================================

@dataclass
class ValidationReport:
    rows: int
    columns: int

    missing_columns: List[str] = field(default_factory=list)
    invalid_text_rows: int = 0

    invalid_label_rows: Dict[str, int] = field(default_factory=dict)
    label_value_violations: Dict[str, int] = field(default_factory=dict)

    notes: Dict[str, Any] = field(default_factory=dict)


# =========================================================
# TASK SCHEMA (TRUTHLENS)
# =========================================================

TASK_SCHEMAS: Dict[str, Dict[str, Any]] = {
    "bias": {
        "required": ["text", "bias"],
        "type": "classification",
        "range": (0, 1),
    },
    "ideology": {
        "required": ["text", "ideology"],
        "type": "classification",
        "range": (0, 2),
    },
    "propaganda": {
        "required": ["text", "propaganda"],
        "type": "classification",
        "range": (0, 1),
    },
    "frame": {
        "required": ["text", "CO", "EC", "HI", "MO", "RE"],
        "type": "multilabel",
        "cols": ["CO", "EC", "HI", "MO", "RE"],
    },
    "narrative": {
        "required": ["text", "hero", "villain", "victim"],
        "type": "multilabel",
        "cols": ["hero", "villain", "victim"],
    },
    "emotion": {
        "required": ["text"] + [f"emotion_{i}" for i in range(20)],
        "type": "multilabel",
        "cols": [f"emotion_{i}" for i in range(20)],
    },
}


# =========================================================
# CORE VALIDATION
# =========================================================

def validate_dataframe(
    df: pd.DataFrame,
    *,
    task: str,
    config: Optional[DataValidatorConfig] = None,
) -> ValidationReport:

    if task not in TASK_SCHEMAS:
        raise ValueError(f"Unknown task: {task}")

    config = config or DataValidatorConfig()
    schema = TASK_SCHEMAS[task]

    report = ValidationReport(
        rows=len(df),
        columns=len(df.columns),
    )

    # -----------------------------------------------------
    # 1. COLUMN CHECK
    # -----------------------------------------------------
    missing = [c for c in schema["required"] if c not in df.columns]
    if missing:
        report.missing_columns = missing
        _handle_error(f"Missing columns: {missing}", config)

    # -----------------------------------------------------
    # 2. TEXT VALIDATION
    # -----------------------------------------------------
    if config.check_text and "text" in df.columns:

        invalid_mask = (
            df["text"].isna()
            | (df["text"].astype(str).str.len() < config.min_text_len)
            | (df["text"].astype(str).str.len() > config.max_text_len)
        )

        report.invalid_text_rows = int(invalid_mask.sum())

        if report.invalid_text_rows > 0:
            _log_warn(f"Invalid text rows: {report.invalid_text_rows}", config)

    # -----------------------------------------------------
    # 3. LABEL VALIDATION
    # -----------------------------------------------------
    if schema["type"] == "classification":
        _validate_classification(df, schema, report, config)

    elif schema["type"] == "multilabel":
        _validate_multilabel(df, schema, report, config)

    logger.info(
        "Validation | task=%s | rows=%d | issues=%d",
        task,
        report.rows,
        report.invalid_text_rows + sum(report.invalid_label_rows.values()),
    )

    return report


# =========================================================
# CLASSIFICATION VALIDATION
# =========================================================

def _validate_classification(df, schema, report, config):

    label_col = schema["required"][1]

    if label_col not in df:
        return

    # invalid (NaN)
    invalid_mask = df[label_col].isna()
    report.invalid_label_rows[label_col] = int(invalid_mask.sum())

    # range check
    if config.enforce_label_range:
        low, high = schema["range"]

        bad_values = ~df[label_col].between(low, high)
        violations = int(bad_values.sum())

        report.label_value_violations[label_col] = violations

        if violations > 0:
            _handle_error(
                f"{label_col} has {violations} values outside [{low}, {high}]",
                config,
            )


# =========================================================
# MULTILABEL VALIDATION
# =========================================================

def _validate_multilabel(df, schema, report, config):

    cols = schema["cols"]

    for col in cols:

        if col not in df:
            continue

        # NaN check
        invalid_mask = df[col].isna()
        report.invalid_label_rows[col] = int(invalid_mask.sum())

        # binary check
        if config.enforce_binary_multilabel:
            bad = ~df[col].isin([0, 1])
            violations = int(bad.sum())

            report.label_value_violations[col] = violations

            if violations > 0:
                _handle_error(
                    f"{col} has non-binary values ({violations} rows)",
                    config,
                )


# =========================================================
# HELPERS
# =========================================================

def _handle_error(msg: str, config: DataValidatorConfig):
    if config.strict:
        raise ValueError(msg)
    else:
        logger.warning(msg)


def _log_warn(msg: str, config: DataValidatorConfig):
    logger.warning(msg)