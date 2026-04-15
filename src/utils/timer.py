"""Timing utilities for logging duration of pipeline steps."""

import time
from collections.abc import Generator
from contextlib import contextmanager

from src.utils.log import log


def _fmt_duration(seconds: float) -> str:
    """Format a duration in seconds as a human-readable string (e.g. '1h 2m 3s').

    Returns a string like '5s', '3m 12s', or '1h 4m 32s'.
    """
    seconds = int(seconds)
    h, remainder = divmod(seconds, 3600)
    m, s = divmod(remainder, 60)
    if h:
        return f"{h}h {m}m {s}s"
    if m:
        return f"{m}m {s}s"
    return f"{s}s"


@contextmanager
def log_timer(label: str) -> Generator[None, None, None]:
    """Context manager that logs how long a block takes with a searchable [TIMER] prefix.

    Usage::

        with log_timer("generation"):
            responses = generate_fn(prompts, prefixes)

    Logs a line like: ``    [TIMER] generation: 1h 4m 32s``
    """
    start = time.monotonic()
    yield
    log(f"    [TIMER] {label}: {_fmt_duration(time.monotonic() - start)}")
