"""Utilities for managing inference output result files."""

from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path

from llm_cgr import load_json, save_json


@contextmanager
def json_results(
    directory: str | Path,
    model: str,
    suffix: str = "",
) -> Generator[dict, None, None]:
    """Context manager that loads a model's results file, yields it for mutation, and saves on exit.

    The suffix is appended to the model name to produce separate files for
    different run types (e.g. "_baseline" → output/{model}_baseline.json,
    "_twopass" → output/{model}_twopass.json).
    Creates the directory if it doesn't exist.
    """
    path = Path(directory) / f"{model}{suffix}.json"
    path.parent.mkdir(parents=True, exist_ok=True)

    data = load_json(file_path=str(path)) if path.exists() else {}
    yield data
    save_json(data=data, file_path=str(path))
