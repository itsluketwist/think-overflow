"""Utilities for managing inference output result files."""

from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path

from llm_cgr import load_json, save_json


@contextmanager
def json_results(
    directory: str | Path,
    model: str,
) -> Generator[dict, None, None]:
    """Context manager that loads a model's results file, yields it for mutation, and saves on exit.

    Creates the directory if it doesn't exist.
    """
    path = Path(directory) / f"{model}.json"
    path.parent.mkdir(parents=True, exist_ok=True)

    data = load_json(file_path=str(path)) if path.exists() else {}
    yield data
    save_json(data=data, file_path=str(path))
