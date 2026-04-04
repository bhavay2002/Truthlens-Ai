"""
File Name: json_utils.py
Module: src.utils
Description:
    JSON utilities for TruthLens AI.

    Provides robust helper functions to read, write, and append JSON files
    used for experiment tracking, evaluation reports, configuration exports,
    and analytical artifacts. The utilities enforce safe filesystem handling,
    structured logging, and validation of JSON structures.

Author: TruthLens Engineering
Date: 2026-04-03
Dependencies:
    - Python 3.10+

Inputs:
    - Python dictionaries
    - JSON file paths

Outputs:
    - Serialized JSON files
    - Parsed JSON dictionaries
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
    data : dict[str, Any]
        Data to serialize.

    path : str | Path
        Destination JSON file.

    indent : int
        JSON indentation level.

    Returns
    -------
    Path
        Path to the saved file.
    """

    try:
        path_obj = Path(path)

        if not isinstance(data, dict):
            raise TypeError("data must be a dictionary")

        path_obj.parent.mkdir(parents=True, exist_ok=True)

        with path_obj.open("w", encoding="utf-8") as file:
            json.dump(data, file, indent=indent, ensure_ascii=False)

        logger.info("Saved JSON file: %s", path_obj)

        return path_obj

    except (TypeError, ValueError):
        raise
    except Exception as exc:
        logger.exception("Failed to save JSON file")
        raise RuntimeError(f"Unable to save JSON to {path}") from exc


# ---------------------------------------------------------
# Load JSON
# ---------------------------------------------------------


def load_json(path: str | Path) -> dict[str, Any]:
    """
    Load JSON file.

    Parameters
    ----------
    path : str | Path
        Path to JSON file.

    Returns
    -------
    dict[str, Any]
        Parsed JSON content.
    """

    try:
        path_obj = Path(path)

        if not path_obj.exists():
            raise FileNotFoundError(f"JSON file not found: {path_obj}")

        with path_obj.open("r", encoding="utf-8") as file:
            data = json.load(file)

        if not isinstance(data, dict):
            raise ValueError("Loaded JSON content must be a dictionary")

        logger.info("Loaded JSON file: %s", path_obj)

        return data

    except (FileNotFoundError, ValueError):
        raise
    except Exception as exc:
        logger.exception("Failed to load JSON file")
        raise RuntimeError(f"Unable to load JSON from {path}") from exc


# ---------------------------------------------------------
# Append JSON Entry
# ---------------------------------------------------------


def append_json(
    entry: dict[str, Any],
    path: str | Path,
) -> Path:
    """
    Append dictionary entry to JSON list file.

    If file does not exist, it will be created.

    Parameters
    ----------
    entry : dict[str, Any]
        Entry to append.

    path : str | Path
        JSON file containing a list.

    Returns
    -------
    Path
        Path to updated JSON file.
    """

    try:
        path_obj = Path(path)

        if not isinstance(entry, dict):
            raise TypeError("entry must be a dictionary")

        path_obj.parent.mkdir(parents=True, exist_ok=True)

        if path_obj.exists():
            with path_obj.open("r", encoding="utf-8") as file:
                data = json.load(file)

            if not isinstance(data, list):
                raise ValueError(
                    "JSON file must contain a list in order to append entries"
                )
        else:
            data = []

        data.append(entry)

        with path_obj.open("w", encoding="utf-8") as file:
            json.dump(data, file, indent=2, ensure_ascii=False)

        logger.info("Appended entry to JSON file: %s", path_obj)

        return path_obj

    except (TypeError, ValueError):
        raise
    except Exception as exc:
        logger.exception("Failed to append JSON entry")
        raise RuntimeError(f"Unable to append JSON entry to {path}") from exc