"""Logging utilities for CLI scripts."""


def log(message: str = "") -> None:
    """Print with immediate flush to ensure output is visible."""
    print(message, flush=True)


def log_header(title: str) -> None:
    """Print a section header."""
    log()
    log("=" * 60)
    log(f"  {title}")
    log("=" * 60)


def log_step(step: int, total: int, message: str) -> None:
    """Print a progress step."""
    log(f"[{step}/{total}] {message}")
