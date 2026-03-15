"""
File: logging_utils.py

Purpose
-------
Logging configuration utilities for TruthLens AI.

This module sets up standardized logging for the entire
application, supporting console and optional file logging.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path


# ---------------------------------------------------------
# Configure Logging
# ---------------------------------------------------------

def configure_logging(
    level: int = logging.INFO,
    log_file: str | Path | None = None,
) -> None:
    """
    Configure application logging.

    Parameters
    ----------
    level : int
        Logging level (e.g., logging.INFO).

    log_file : str | Path | None
        Optional log file path.
    """

    root_logger = logging.getLogger()
    root_logger.setLevel(level)

    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    has_stream_handler = any(
        isinstance(handler, logging.StreamHandler)
        for handler in root_logger.handlers
    )
    if not has_stream_handler:
        stream_handler = logging.StreamHandler(sys.stdout)
        stream_handler.setFormatter(formatter)
        root_logger.addHandler(stream_handler)

    if log_file is not None:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        resolved_log_path = log_path.resolve()

        has_matching_file_handler = any(
            isinstance(handler, logging.FileHandler)
            and Path(handler.baseFilename).resolve() == resolved_log_path
            for handler in root_logger.handlers
        )

        if not has_matching_file_handler:
            file_handler = logging.FileHandler(log_path, encoding="utf-8")
            file_handler.setFormatter(formatter)
            root_logger.addHandler(file_handler)
