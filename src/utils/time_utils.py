"""
File Name: time_utils.py
Module: src.utils
Description:
    Time utilities for TruthLens AI.

    Provides helpers for generating timestamps, retrieving current
    datetime values, measuring runtime for functions, and supporting
    experiment timing utilities used in training, evaluation, and
    experiment tracking pipelines.

Author: TruthLens Engineering
Date: 2026-04-03
Dependencies:
    - Python 3.10+

Inputs:
    - Callable functions
    - Runtime parameters

Outputs:
    - Timestamp strings
    - datetime objects
    - execution runtime measurements
"""

from __future__ import annotations

import logging
import time
from datetime import datetime, timezone
from typing import Callable, Any, Tuple, TypeVar


# ---------------------------------------------------------
# Logging
# ---------------------------------------------------------

logger = logging.getLogger(__name__)


# ---------------------------------------------------------
# Type Definitions
# ---------------------------------------------------------

T = TypeVar("T")


# ---------------------------------------------------------
# Timestamp Generator
# ---------------------------------------------------------


def timestamp() -> str:
    """
    Return formatted timestamp string.

    Returns
    -------
    str
        Timestamp in format YYYY-MM-DD_HH-MM-SS (UTC)
    """

    return datetime.now(timezone.utc).strftime("%Y-%m-%d_%H-%M-%S")


# ---------------------------------------------------------
# Current Datetime
# ---------------------------------------------------------


def current_datetime() -> datetime:
    """
    Return current datetime object (UTC).

    Returns
    -------
    datetime
    """

    return datetime.now(timezone.utc)


# ---------------------------------------------------------
# Runtime Measurement
# ---------------------------------------------------------


def measure_runtime(
    func: Callable[..., T],
    *args: Any,
    **kwargs: Any,
) -> Tuple[T, float]:
    """
    Measure execution time of a function.

    Parameters
    ----------
    func : Callable
        Function to execute.

    Returns
    -------
    Tuple[T, float]
        (result, runtime_seconds)
    """

    start_time = time.perf_counter()

    try:
        result = func(*args, **kwargs)
    except Exception as exc:
        logger.exception(
            "Error occurred during execution of '%s'",
            getattr(func, "__name__", "unknown_function"),
        )
        raise exc

    end_time = time.perf_counter()

    runtime = end_time - start_time

    logger.info(
        "Function '%s' executed in %.6f seconds",
        getattr(func, "__name__", "unknown_function"),
        runtime,
    )

    return result, runtime


# ---------------------------------------------------------
# Simple Timer Utility
# ---------------------------------------------------------


class Timer:
    """
    Lightweight timer utility for measuring code blocks.

    Example
    -------
    timer = Timer()
    timer.start()
    ...
    elapsed = timer.stop()
    """

    def __init__(self) -> None:
        self._start: float | None = None

    def start(self) -> None:
        """Start timer."""
        self._start = time.perf_counter()

    def stop(self) -> float:
        """
        Stop timer and return elapsed seconds.

        Returns
        -------
        float
            Elapsed runtime in seconds.
        """

        if self._start is None:
            raise RuntimeError("Timer was not started")

        elapsed = time.perf_counter() - self._start

        logger.debug("Timer stopped: %.6f seconds", elapsed)

        self._start = None

        return elapsed