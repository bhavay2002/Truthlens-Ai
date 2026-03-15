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

    # Prevent duplicate handlers if logging already configured
    if root_logger.handlers:
        return

    handlers: list[logging.Handler] = [
        logging.StreamHandler(sys.stdout)
    ]

    if log_file is not None:

        log_path = Path(log_file)

        log_path.parent.mkdir(parents=True, exist_ok=True)

        handlers.insert(
            0,
            logging.FileHandler(log_path, encoding="utf-8")
        )

    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=handlers,
    )