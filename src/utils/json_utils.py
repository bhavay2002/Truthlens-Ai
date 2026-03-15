"""
File: json_utils.py

Purpose
-------
JSON utilities for TruthLens AI.

Provides helper functions to read and write JSON files used
for experiment tracking, evaluation reports, and data analysis.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any


# ---------------------------------------------------------
# Logging
# ---------------------------------------------------------

logger = logging.getLogger(__name__)


# ---------------------------------------------------------
# Save JSON
# ---------------------------------------------------------

def save_json(
    data: dict[str, Any],
    path: str | Path,
    indent: int = 2,
) -> Path:
    """
    Save dictionary to JSON file.

    Parameters
    ----------
    data : dict
        Data to serialize.

    path : str | Path
        Destination file.

    indent : int
        JSON indentation level.

    Returns
    -------
    Path
        Saved file path.
    """

    try:

        path = Path(path)

        path.parent.mkdir(parents=True, exist_ok=True)

        with path.open("w", encoding="utf-8") as f:

            json.dump(data, f, indent=indent)

        logger.info("Saved JSON file: %s", path)

        return path

    except Exception:

        logger.exception("Failed to save JSON")

        raise


# ---------------------------------------------------------
# Load JSON
# ---------------------------------------------------------

def load_json(path: str | Path) -> dict[str, Any]:
    """
    Load JSON file.

    Parameters
    ----------
    path : str | Path

    Returns
    -------
    dict
    """

    try:

        path = Path(path)

        if not path.exists():

            raise FileNotFoundError(f"JSON file not found: {path}")

        with path.open("r", encoding="utf-8") as f:

            data = json.load(f)

        logger.info("Loaded JSON file: %s", path)

        return data

    except Exception:

        logger.exception("Failed to load JSON")

        raise


# ---------------------------------------------------------
# Append JSON Entry
# ---------------------------------------------------------

def append_json(
    entry: dict[str, Any],
    path: str | Path,
) -> Path:
    """
    Append dictionary entry to JSON list file.

    If file does not exist, it is created.

    Parameters
    ----------
    entry : dict
        Entry to append.

    path : str | Path

    Returns
    -------
    Path
    """

    try:

        path = Path(path)

        path.parent.mkdir(parents=True, exist_ok=True)

        if path.exists():

            with path.open("r", encoding="utf-8") as f:

                data = json.load(f)

            if not isinstance(data, list):

                raise ValueError("JSON file must contain a list to append entries")

        else:

            data = []

        data.append(entry)

        with path.open("w", encoding="utf-8") as f:

            json.dump(data, f, indent=2)

        logger.info("Appended entry to JSON file: %s", path)

        return path

    except Exception:

        logger.exception("Failed to append JSON entry")

        raise