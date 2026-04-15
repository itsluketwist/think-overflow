"""Configuration file loading utilities."""

from typing import Any

import yaml


def _load_yaml(config_file: str) -> dict[str, Any]:
    """Load and parse a YAML config file.

    Returns the parsed dict, or an empty dict if the file is blank.
    """
    with open(file=config_file) as f:
        return yaml.safe_load(stream=f) or {}


def load_yaml_config(
    config_file: str,
    key: str,
) -> dict[str, Any]:
    """Load a specific key from a yaml config file.

    Returns the config dict for the given key.
    Raises KeyError if the key is not found.
    """
    config = _load_yaml(config_file=config_file)

    if key in config:
        return config[key]

    raise KeyError(
        f"key '{key}' not found in {config_file}",
    )
