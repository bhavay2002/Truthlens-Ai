"""
File: helper_functions.py

Purpose
-------
Utility helper functions used across TruthLens AI.

This module contains small reusable helpers for filesystem
operations and other common tasks.
"""

import logging
from pathlib import Path


# ---------------------------------------------------------
# Logging
# ---------------------------------------------------------

logger = logging.getLogger(__name__)


# ---------------------------------------------------------
# Create Folder
# ---------------------------------------------------------

def create_folder(path: str | Path) -> Path:
    """
    Create directory if it does not exist.

    Parameters
    ----------
    path : str | Path
        Directory path.

    Returns
    -------
    Path
        Created or existing directory path.
    """

    try:

        path_obj = Path(path)

        path_obj.mkdir(parents=True, exist_ok=True)

        logger.debug("Directory ensured: %s", path_obj)

        return path_obj

    except Exception:

        logger.exception("Failed to create directory")

        raise