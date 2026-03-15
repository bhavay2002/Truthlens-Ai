"""
File: time_utils.py

Purpose
-------
Time utilities for TruthLens AI.

Provides helpers for generating timestamps and measuring runtime
for training, evaluation, and experiment tracking.
"""

from __future__ import annotations

import logging
import time
from datetime import datetime
from typing import Callable, Any


# ---------------------------------------------------------
# Logging
# ---------------------------------------------------------

logger = logging.getLogger(__name__)


# ---------------------------------------------------------
# Timestamp Generator
# ---------------------------------------------------------

def timestamp() -> str:
    """
    Return formatted timestamp string.

    Returns
    -------
    str
        Timestamp in format YYYY-MM-DD_HH-MM-SS
    """

    return datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


# ---------------------------------------------------------
# Current Datetime
# ---------------------------------------------------------

def current_datetime() -> datetime:
    """
    Return current datetime object.
    """

    return datetime.now()


# ---------------------------------------------------------
# Runtime Measurement
# ---------------------------------------------------------

def measure_runtime(func: Callable[..., Any], *args, **kwargs) -> tuple[Any, float]:
    """
    Measure execution time of a function.

    Parameters
    ----------
    func : Callable

    Returns
    -------
    tuple
        (result, runtime_seconds)
    """

    start_time = time.time()

    result = func(*args, **kwargs)

    end_time = time.time()

    runtime = end_time - start_time

    logger.info("Function '%s' executed in %.3f seconds", func.__name__, runtime)

    return result, runtime