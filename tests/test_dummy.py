"""Smoke tests to verify the package imports and config loading work correctly."""

from src.utils.config import load_yaml_config


def test_load_model_config():
    # check that the model config file can be read and contains expected keys
    config = load_yaml_config(
        config_file="config/models.yaml",
        key="qwen3-8b",
    )
    assert "model_path" in config


def test_load_inference_config():
    # check that the inference config file can be read and contains expected keys
    config = load_yaml_config(
        config_file="config/inference.yaml",
        key="greedy",
    )
    assert "temperature" in config
