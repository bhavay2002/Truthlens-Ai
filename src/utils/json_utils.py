from __future__ import annotations

import json
import logging
import os
import tempfile
from pathlib import Path
from typing import Any
from contextlib import contextmanager
from collections import OrderedDict
from threading import Lock

logger = logging.getLogger(__name__)

# =========================================================
# LOCK MANAGEMENT
# =========================================================

# Max distinct file locks to retain
_MAX_LOCKS = 1024

_FILE_LOCKS: "OrderedDict[str, Lock]" = OrderedDict()
_LOCKS_GUARD = Lock()


def _normalize_path(path: Path) -> str:
    return str(path.resolve())


def _get_lock(path: Path) -> Lock:
    key = _normalize_path(path)

    with _LOCKS_GUARD:
        lock = _FILE_LOCKS.get(key)

        if lock is not None:
            # mark as recently used
            _FILE_LOCKS.move_to_end(key)
            return lock

        # create new lock
        lock = Lock()
        _FILE_LOCKS[key] = lock

        # enforce LRU cap
        if len(_FILE_LOCKS) > _MAX_LOCKS:
            _FILE_LOCKS.popitem(last=False)

        return lock


@contextmanager
def _locked_path(path: Path):
    lock = _get_lock(path)
    with lock:
        yield


# =========================================================
# ATOMIC WRITE
# =========================================================

def _atomic_write_json(path_obj: Path, data: Any, indent: int = 2) -> None:
    path_obj.parent.mkdir(parents=True, exist_ok=True)

    tmp_path = None

    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path_obj.parent,
            delete=False,
        ) as tmp:
            json.dump(data, tmp, indent=indent, ensure_ascii=False)
            tmp.flush()
            os.fsync(tmp.fileno())  # ensure durability
            tmp_path = Path(tmp.name)

        os.replace(tmp_path, path_obj)

    finally:
        if tmp_path and tmp_path.exists():
            try:
                tmp_path.unlink(missing_ok=True)
            except Exception:
                logger.warning("Failed to cleanup temp file: %s", tmp_path)


# =========================================================
# VALIDATION
# =========================================================

def _validate_json_serializable(data: Any) -> None:
    try:
        json.dumps(data)
    except (TypeError, ValueError) as exc:
        raise ValueError("Data is not JSON serializable") from exc


# =========================================================
# SAVE JSON
# =========================================================

def save_json(
    data: dict[str, Any],
    path: str | Path,
    indent: int = 2,
) -> Path:

    path_obj = Path(path)

    if not isinstance(data, dict):
        raise TypeError("data must be a dictionary")

    _validate_json_serializable(data)

    try:
        with _locked_path(path_obj):
            _atomic_write_json(path_obj, data, indent=indent)

        logger.info("Saved JSON file: %s", path_obj)
        return path_obj

    except Exception as exc:
        logger.exception("Failed to save JSON file")
        raise RuntimeError(f"Unable to save JSON to {path_obj}") from exc


# =========================================================
# LOAD JSON
# =========================================================

def load_json(path: str | Path) -> dict[str, Any]:

    path_obj = Path(path)

    if not path_obj.exists():
        raise FileNotFoundError(f"JSON file not found: {path_obj}")

    try:
        with path_obj.open("r", encoding="utf-8") as file:
            data = json.load(file)

        if not isinstance(data, dict):
            raise ValueError("Loaded JSON content must be a dictionary")

        logger.info("Loaded JSON file: %s", path_obj)
        return data

    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON format in {path_obj}") from exc
    except Exception as exc:
        logger.exception("Failed to load JSON file")
        raise RuntimeError(f"Unable to load JSON from {path_obj}") from exc


# =========================================================
# APPEND JSON (JSONL FORMAT)
# =========================================================

def append_json(
    entry: dict[str, Any],
    path: str | Path,
) -> Path:
    """
    Append entry as JSONL (one JSON object per line).
    """

    path_obj = Path(path)

    if not isinstance(entry, dict):
        raise TypeError("entry must be a dictionary")

    _validate_json_serializable(entry)

    try:
        with _locked_path(path_obj):
            path_obj.parent.mkdir(parents=True, exist_ok=True)

            with path_obj.open("a", encoding="utf-8") as file:
                file.write(json.dumps(entry, ensure_ascii=False) + "\n")
                file.flush()
                os.fsync(file.fileno())

        logger.info("Appended JSONL entry: %s", path_obj)
        return path_obj

    except Exception as exc:
        logger.exception("Failed to append JSON entry")
        raise RuntimeError(f"Unable to append JSON entry to {path_obj}") from exc