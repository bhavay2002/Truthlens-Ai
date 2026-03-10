from __future__ import annotations

from pathlib import Path

from src.models.model_utils import load_model, save_model
from src.utils.helper_functions import create_folder
from src.utils.settings import load_settings


def test_load_settings_has_expected_types():
    settings = load_settings()

    assert isinstance(settings.model.path, Path)
    assert isinstance(settings.paths.training_log_path, Path)
    assert settings.training.cross_validation_splits >= 2
    assert settings.training.optuna_trials >= 1


def test_create_folder_returns_path(tmp_path: Path):
    target = tmp_path / "nested" / "folder"
    created = create_folder(target)

    assert created == target
    assert target.exists()


def test_model_utils_accept_path_objects(tmp_path: Path):
    model_file = tmp_path / "model.joblib"
    payload = {"a": 1, "b": [1, 2, 3]}

    save_path = save_model(payload, model_file)
    loaded = load_model(save_path)

    assert save_path == model_file
    assert loaded == payload
