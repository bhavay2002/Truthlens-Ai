from __future__ import annotations

from src.utils import settings as settings_module


def test_load_settings_supports_legacy_cv_and_tuning_sections(monkeypatch):
    legacy_config = {
        "model": {"path": "models/roberta_model"},
        "training": {
            "text_column": "engineered_text",
            "seed": 7,
            "epochs": 2,
            "batch_size": 8,
            "learning_rate": 2e-5,
            "validation_size": 0.2,
            "test_size": 0.2,
        },
        "cross_validation": {
            "enabled": True,
            "splits": 3,
            "metric": "accuracy",
        },
        "hyperparameter_tuning": {
            "enabled": True,
            "trials": 4,
            "direction": "maximize",
            "metric": "f1",
            "learning_rate_range": [1e-6, 2e-5],
            "batch_sizes": [4, 8],
            "epoch_choices": [2, 4],
            "validation_split": 0.25,
        },
    }

    settings_module.load_settings.cache_clear()
    monkeypatch.setattr(settings_module, "load_config", lambda: legacy_config)
    settings = settings_module.load_settings()

    assert settings.training.run_cross_validation is True
    assert settings.training.cross_validation_splits == 3
    assert settings.training.cross_validation_metric == "accuracy"

    assert settings.training.run_hyperparameter_tuning is True
    assert settings.training.optuna_trials == 4
    assert settings.training.optuna_direction == "maximize"
    assert settings.training.optuna_metric == "f1"
    assert settings.training.optuna_learning_rate_min == 1e-6
    assert settings.training.optuna_learning_rate_max == 2e-5
    assert settings.training.optuna_batch_sizes == (4, 8)
    assert settings.training.optuna_epoch_choices == (2, 4)
    assert settings.training.optuna_validation_split == 0.25

    settings_module.load_settings.cache_clear()
