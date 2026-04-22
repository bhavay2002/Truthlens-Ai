"""
File Name: logging_utils.py
Module: src.utils
Description:
    Logging configuration utilities for TruthLens AI.

    This module provides centralized logging configuration for the
    entire application. It supports console logging, optional file
    logging, structured formatting, and safeguards against duplicate
    handlers.

    Designed for use across training pipelines, inference services,
    and evaluation scripts.

Author: TruthLens Engineering
Date: 2026-04-03
Dependencies:
    - Python 3.10+

Inputs:
    - Logging level
    - Optional log file path

Outputs:
    - Configured Python logging system
"""

from __future__ import annotations

import logging
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Optional


# ---------------------------------------------------------
# Logging Formatter
# ---------------------------------------------------------


def _create_formatter() -> logging.Formatter:
    """
    Create standardized logging formatter.

    Returns
    -------
    logging.Formatter
    """

    return logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


# ---------------------------------------------------------
# Configure Logging
# ---------------------------------------------------------


def configure_logging(
    level: int = logging.INFO,
    log_file: Optional[str | Path] = None,
) -> None:
    """
    Configure application-wide logging.

    Parameters
    ----------
    level : int
        Logging level (e.g., logging.INFO).

    log_file : Optional[str | Path]
        Optional file path for persistent logs.
    """

    root_logger = logging.getLogger()
    root_logger.setLevel(level)

    formatter = _create_formatter()

    _ensure_stream_handler(root_logger, formatter)

    if log_file is not None:
        _ensure_file_handler(root_logger, formatter, log_file)


# ---------------------------------------------------------
# Stream Handler
# ---------------------------------------------------------


def _ensure_stream_handler(
    logger: logging.Logger,
    formatter: logging.Formatter,
) -> None:
    """
    Ensure console logging handler exists.

    Parameters
    ----------
    logger : logging.Logger
    formatter : logging.Formatter
    """

    has_stream_handler = any(
        isinstance(handler, logging.StreamHandler)
        and not isinstance(handler, logging.FileHandler)
        for handler in logger.handlers
    )

    if not has_stream_handler:
        stream_handler = logging.StreamHandler(sys.stdout)
        stream_handler.setFormatter(formatter)

        logger.addHandler(stream_handler)


# ---------------------------------------------------------
# File Handler
# ---------------------------------------------------------


def _ensure_file_handler(
    logger: logging.Logger,
    formatter: logging.Formatter,
    log_file: str | Path,
) -> None:
    """
    Ensure file logging handler exists.

    Parameters
    ----------
    logger : logging.Logger
    formatter : logging.Formatter
    log_file : str | Path
    """

    log_path = Path(log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    resolved_log_path = log_path.resolve()

    has_matching_file_handler = any(
        isinstance(handler, logging.FileHandler)
        and Path(handler.baseFilename).resolve() == resolved_log_path
        for handler in logger.handlers
    )

    if not has_matching_file_handler:
        # m3: rotate to bound disk usage; delay=True so the file isn't created
        # until the first record is written.
        file_handler = RotatingFileHandler(
            log_path,
            maxBytes=50 * 1024 * 1024,
            backupCount=5,
            encoding="utf-8",
            delay=True,
        )
        file_handler.setFormatter(formatter)

        logger.addHandler(file_handler)