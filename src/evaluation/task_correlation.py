"""
File Name: task_correlation.py
Module: TruthLens AI - Task Correlation
Description:
    Utilities for analyzing statistical relationships between predictions of
    multiple tasks in the TruthLens multi-task system. Computes correlation
    matrices across task outputs and supports exporting the matrix for
    reporting, diagnostics, and research analysis.
Dependencies:
    numpy
    pandas
    pathlib
    logging
    typing
Inputs:
    predictions: dictionary or dataframe containing task predictions
    path: file path where the correlation matrix should be saved
Outputs:
    pandas DataFrame representing the task correlation matrix
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Any

import pandas as pd


logger = logging.getLogger(__name__)


def _validate_predictions(predictions: Dict[str, Any] | pd.DataFrame) -> pd.DataFrame:
    """
    Validate and convert predictions into a DataFrame.
    """

    if isinstance(predictions, pd.DataFrame):
        df = predictions.copy()
    elif isinstance(predictions, dict):
        df = pd.DataFrame(predictions)
    else:
        raise TypeError(
            "predictions must be a dictionary or pandas DataFrame."
        )

    if df.empty:
        raise ValueError("Prediction data cannot be empty.")

    if df.shape[1] < 2:
        raise ValueError(
            "At least two tasks are required to compute correlations."
        )

    return df


def compute_task_correlation(
    predictions: Dict[str, Any] | pd.DataFrame
) -> pd.DataFrame:
    """
    Compute correlation matrix between task predictions.
    """

    df = _validate_predictions(predictions)

    logger.info("Computing task correlation matrix")

    try:
        df_num = df.apply(pd.to_numeric, errors="coerce")
        usable = df_num.dropna(axis=1, how="all")
        if usable.shape[1] < 2:
            raise ValueError(
                "Need at least two numeric task columns for correlation."
            )
        corr = usable.corr(method="pearson")
    except Exception as exc:
        logger.exception("Failed to compute correlation matrix")
        raise RuntimeError("Correlation computation failed") from exc

    return corr


def save_correlation_matrix(
    corr: pd.DataFrame,
    path: str | Path
) -> Path:
    """
    Save correlation matrix to CSV.
    """

    if not isinstance(corr, pd.DataFrame):
        raise TypeError("corr must be a pandas DataFrame.")

    output_path = Path(path)

    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        corr.to_csv(output_path)
        logger.info("Correlation matrix saved to %s", output_path)
    except Exception as exc:
        logger.exception("Failed to save correlation matrix")
        raise RuntimeError("Saving correlation matrix failed") from exc

    return output_path
