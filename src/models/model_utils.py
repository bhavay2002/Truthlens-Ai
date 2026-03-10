from pathlib import Path
from typing import Any

import joblib


def save_model(model: Any, path: str | Path) -> Path:
    path_obj = Path(path)
    path_obj.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path_obj)
    return path_obj


def load_model(path: str | Path) -> Any:
    return joblib.load(Path(path))
