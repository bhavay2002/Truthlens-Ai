from src.utils.config_loader import load_config


def test_config_loading():

    config = load_config("config/config.yaml")

    assert "model" in config
    assert "training" in config
    assert "data" in config