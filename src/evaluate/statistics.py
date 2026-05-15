"""Reasoning statistics computed over evaluation results."""

import math
from statistics import NormalDist
from typing import Any

import thinkpack
from thinkpack import ParsedResponse


def wilson_ci(
    count: int,
    total: int,
    confidence: float = 0.95,
) -> list[float]:
    """Compute Wilson score confidence interval for a proportion.

    Handles proportions near 0 or 1 correctly without scipy.
    Returns [0.0, 1.0] when total is 0 (no data).

    Returns [lower, upper] as floats.
    """
    if total == 0:
        return [0.0, 1.0]
    p = count / total
    # z-score derived from the standard normal distribution for the given confidence level
    z = NormalDist().inv_cdf(1 - (1 - confidence) / 2)
    z2 = z * z
    # standard wilson score formula
    center = (p + z2 / (2 * total)) / (1 + z2 / total)
    margin = (z * math.sqrt(p * (1 - p) / total + z2 / (4 * total**2))) / (
        1 + z2 / total
    )
    return [max(0.0, center - margin), min(1.0, center + margin)]


def compute_pass_at_1(
    counts: list[int],
    total: int,
    k: int,
) -> tuple[list[float], float, list[float]]:
    """Compute per-sample-index pass rates, their average, and a 95% CI.

    The CI denominator is k * total: each of the k sample indices contributes `total` trials.

    Returns (rate_list, average_rate, ci).
    """
    rate_list = [c / total for c in counts] if total else [0.0] * k
    avg = sum(rate_list) / len(rate_list) if rate_list else 0.0
    ci = wilson_ci(count=sum(counts), total=k * total)
    return rate_list, avg, ci


def compute_reasoning_statistics(
    parsed_responses: list[list[ParsedResponse]],
    details: list[list[dict]] | None,
    results_breakdown: list[dict[str, Any]],
) -> dict[str, Any]:
    """Compute reasoning reliability and overflow statistics.

    Uses thinkpack.compute_stats for reasoning quality metrics (macro-averaged
    across tasks). Derives transition_rate and overflow_rate from the details
    structure (the unified per-sample dict stored alongside each response).

    transition_rate naturally reflects the run mode: it is always 0 for onepass
    (transition=False is stored for every sample) and meaningful for twopass.

    Returns a flat dict of scalar rates — no per-sample-index lists.
    """
    from src.utils.log import log

    empty = {
        "valid_reasoning_rate": 0.0,
        "invalid_reasoning_rate": 0.0,
        "missing_reasoning_rate": 0.0,
        "truncated_reasoning_rate": 0.0,
        "empty_reasoning_rate": 0.0,
        "answer_rate": 0.0,
        "transition_rate": 0.0,
        "overflow_rate": 0.0,
        "answer_extracted_rate": 0.0,
    }

    if not parsed_responses:
        return empty

    # thinkpack computes reliability metrics macro-averaged across tasks
    tp = thinkpack.compute_stats(responses=parsed_responses)

    # transition and overflow rates from the details structure
    transition_rate = 0.0
    overflow_rate = 0.0
    if details:
        flat = [s for task in details for s in task]
        n = len(flat)
        if n:
            transition_rate = sum(s["transition"] for s in flat) / n
            overflow_rate = sum(s["truncated"] for s in flat) / n

    # answer extraction rate from the per-sample evaluator breakdown
    flat_samples = [s for task in results_breakdown for s in task["samples"]]
    n_samples = len(flat_samples)
    answer_extracted_rate = (
        sum(1 for s in flat_samples if s.get("answer_extracted", True)) / n_samples
        if n_samples
        else 0.0
    )

    log(
        f"    Reasoning: valid={tp.valid_reasoning_rate:.1%}, "
        f"missing={tp.missing_reasoning_rate:.1%}, "
        f"transition={transition_rate:.1%}, overflow={overflow_rate:.1%}, "
        f"extracted={answer_extracted_rate:.1%}",
    )

    return {
        # reasoning reliability from thinkpack (macro-averaged across tasks)
        "valid_reasoning_rate": tp.valid_reasoning_rate,
        "invalid_reasoning_rate": tp.invalid_reasoning_rate,
        "missing_reasoning_rate": tp.missing_reasoning_rate,
        "truncated_reasoning_rate": tp.truncated_reasoning_rate,
        "empty_reasoning_rate": tp.empty_reasoning_rate,
        "answer_rate": tp.answer_rate,
        # our custom stats derived from the details structure
        "transition_rate": transition_rate,
        "overflow_rate": overflow_rate,
        "answer_extracted_rate": answer_extracted_rate,
    }


def aggregate_token_counts(details: list[list[dict]]) -> dict:
    """Compute mean token statistics from a details list.

    Reads prompt_tokens_1/2 and completion_tokens_1/2 from each sample dict.
    Pass 2 keys are None for onepass runs and are excluded from their means.

    Returns a dict of mean token stats, or {} if details is empty.
    """
    flat = [s for task in details for s in task]
    if not flat:
        return {}
    n = len(flat)

    def mean_optional(key: str) -> float | None:
        # exclude None values (onepass has no pass 2)
        vals = [s[key] for s in flat if s.get(key) is not None]
        return sum(vals) / len(vals) if vals else None

    return {
        "mean_prompt_tokens_1": sum(s["prompt_tokens_1"] for s in flat) / n,
        "mean_completion_tokens_1": sum(s["completion_tokens_1"] for s in flat) / n,
        "mean_prompt_tokens_2": mean_optional("prompt_tokens_2"),
        "mean_completion_tokens_2": mean_optional("completion_tokens_2"),
    }
