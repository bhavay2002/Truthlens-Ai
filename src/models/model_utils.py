import re
from pathlib import Path
from typing import Any

import joblib


def save_model(model: Any, path: str | Path) -> Path:
    """
    Save a trained model to disk.

    This function serializes a machine learning model using joblib
    and ensures that the target directory exists before saving.

    Parameters
    ----------
    model : Any
        The trained model object to be saved.
    path : str | Path
        Destination file path where the model will be stored.

    Returns
    -------
    Path
        The resolved path where the model was saved.
    """

    path_obj = Path(path)

    # Ensure directory exists
    path_obj.parent.mkdir(parents=True, exist_ok=True)

    # Serialize the model using joblib
    joblib.dump(model, path_obj)

    return path_obj


def load_model(path: str | Path) -> Any:
    """
    Load a serialized model from disk.

    Parameters
    ----------
    path : str | Path
        Path to the saved model file.

    Returns
    -------
    Any
        The deserialized model object.
    """

    path_obj = Path(path)

    # Load and return the model
    return joblib.load(path_obj)


def preprocess_text(text: str) -> str:
    """
    Basic text normalization for inference.
    """
    normalized = str(text).replace("\n", " ").replace("\t", " ")
    normalized = re.sub(r"\s+", " ", normalized).strip()
    return normalized
