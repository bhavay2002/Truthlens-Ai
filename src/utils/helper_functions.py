"""
File Name: helper_functions.py
Module: src.utils
Description:
    General-purpose helper utilities used across the TruthLens AI system.

    This module provides small reusable utilities for filesystem management,
    safe directory creation, file validation, and common path operations.
    All functions are designed to be robust, reusable, and safe for
    production ML pipelines.

Author: TruthLens Engineering
Date: 2026-04-03
Dependencies:
    - Python 3.10+

Inputs:
    - File paths
    - Directory paths

Outputs:
    - Validated Path objects
    - Created directories
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable


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

    Raises
    ------
    RuntimeError
        If directory creation fails.
    """

    try:
        path_obj = Path(path)
        path_obj.mkdir(parents=True, exist_ok=True)

        logger.debug("Directory ensured: %s", path_obj)

        return path_obj

    except Exception as exc:
        logger.exception("Failed to create directory: %s", path)
        raise RuntimeError(f"Unable to create directory: {path}") from exc


# ---------------------------------------------------------
# Ensure Multiple Folders
# ---------------------------------------------------------


def ensure_directories(paths: Iterable[str | Path]) -> list[Path]:
    """
    Ensure multiple directories exist.

    Parameters
    ----------
    paths : Iterable[str | Path]
        List of directory paths.

    Returns
    -------
    list[Path]
        List of created/validated directory paths.
    """

    created_paths: list[Path] = []

    for path in paths:
        created_paths.append(create_folder(path))

    return created_paths


# ---------------------------------------------------------
# Validate File Exists
# ---------------------------------------------------------


def ensure_file_exists(path: str | Path) -> Path:
    """
    Validate that a file exists.

    Parameters
    ----------
    path : str | Path
        File path.

    Returns
    -------
    Path
        Validated file path.

    Raises
    ------
    FileNotFoundError
        If file does not exist.
    """

    path_obj = Path(path)

    if not path_obj.exists() or not path_obj.is_file():
        logger.error("File does not exist: %s", path_obj)
        raise FileNotFoundError(f"File not found: {path_obj}")

    return path_obj


# ---------------------------------------------------------
# Safe Path Conversion
# ---------------------------------------------------------


def to_path(path: str | Path) -> Path:
    """
    Convert string or Path-like object to Path.

    Parameters
    ----------
    path : str | Path

    Returns
    -------
    Path
    """

    if isinstance(path, Path):
        return path

    return Path(path)


# ---------------------------------------------------------
# File Size Utility
# ---------------------------------------------------------


def get_file_size(path: str | Path) -> int:
    """
    Get file size in bytes.

    Parameters
    ----------
    path : str | Path

    Returns
    -------
    int
        File size in bytes.
    """

    file_path = ensure_file_exists(path)

    return file_path.stat().st_size