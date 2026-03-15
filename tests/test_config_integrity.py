from src.utils.settings import load_settings


def test_config_integrity():

    config = load_settings()

    assert config.training.batch_size > 0
    assert config.training.epochs >= 1
    assert config.model.path is not None