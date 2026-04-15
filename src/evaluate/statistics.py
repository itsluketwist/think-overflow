"""Reasoning statistics computed over evaluation results."""

import math
from statistics import NormalDist
from typing import Any

from src.evaluate.parser import ParsedResponse


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


def _is_correct(sample: dict[str, Any]) -> bool:
    """Return True if the sample is correct, checking both 'correct' and 'passed' keys."""
    if "correct" in sample:
        return sample["correct"]
    if "passed" in sample:
        return sample["passed"]
    return False


def _is_correct_anywhere(sample: dict[str, Any]) -> bool:
    """Return True if the sample is correct in the answer section OR the reasoning block."""
    return _is_correct(sample) or bool(sample.get("correct_in_reasoning"))


def _compute_pass_at_1(
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
    results_breakdown: list[dict[str, Any]],
    truncated_flags: list[list[bool]] | None = None,
    two_pass: bool = False,
) -> dict[str, Any]:
    """Compute reasoning token rates and conditional accuracy metrics in a single pass.

    Each metric is returned both as a scalar (averaged across sample indices) and as a
    list (one entry per sample index). k is taken from the first task; tasks with fewer
    samples only contribute to the indices they have.

    two_pass controls how overflow is detected:
    - two_pass=True: truncated_flags ([task][sample] booleans) holds pass 1 cap hits;
      these become reasoning_truncation_rate (two-pass only). is_overflow from the pass 2
      finish reason gives answer_overflow_rate. has_truncated_reasoning is always False
      in two-pass because _build_pass2_prefix always injects </think>.
    - two_pass=False (baseline): truncated_flags holds per-sample finish_reason=="length"
      flags; has_truncated_reasoning splits into reasoning_overflow_rate (overflow inside
      the reasoning block) vs answer_overflow_rate (overflow after reasoning completed).

    Returns a dict with keys: reasoning_block, any_reasoning, valid_reasoning,
    reasoning_truncation_rate (two-pass only), reasoning_overflow_rate (baseline only),
    answer_overflow_rate (both), reasoning_accuracy, reasoning_leak_accuracy
    (each as a scalar and a _list variant; reasoning_accuracy(_list) may be None/contain None).
    """
    from src.utils.log import log

    if not parsed_responses:
        return {
            "reasoning_block": 0.0,
            "reasoning_block_list": [],
            "any_reasoning": 0.0,
            "any_reasoning_list": [],
            "valid_reasoning": 0.0,
            "valid_reasoning_list": [],
            "reasoning_truncation_rate": 0.0,
            "reasoning_truncation_rate_list": [],
            "reasoning_overflow_rate": 0.0,
            "reasoning_overflow_rate_list": [],
            "answer_overflow_rate": 0.0,
            "answer_overflow_rate_list": [],
            "answer_extracted_rate": 0.0,
            "answer_extracted_rate_list": [],
            "reasoning_accuracy": None,
            "reasoning_accuracy_list": [],
            "reasoning_leak_accuracy": None,
            "reasoning_leak_accuracy_list": [],
        }

    # k is taken from the first task; tasks with fewer samples contribute to fewer indices
    k = len(parsed_responses[0])
    total_tasks = len(parsed_responses)

    # per-sample-index accumulators
    block_counts = [0] * k
    any_reasoning_counts = [0] * k
    valid_reasoning_counts = [0] * k
    # two-pass only: pass 1 hit the max_think_tokens cap
    reasoning_truncation_counts = [0] * k
    # baseline only: single-pass overflowed while still inside the reasoning block
    reasoning_overflow_counts = [0] * k
    # both modes: final generation hit the token limit (pass 2 or baseline answer)
    answer_overflow_counts = [0] * k
    extracted_counts = [0] * k
    # for reasoning_accuracy: track correct and total per sample index
    correct_reasoning_counts = [0] * k
    total_reasoning_counts = [0] * k
    # for leak accuracy: correct per sample index (denominator is total_tasks)
    correct_leak_counts = [0] * k

    for task_idx, (task_parsed, task_breakdown) in enumerate(
        zip(parsed_responses, results_breakdown),
    ):
        samples = task_breakdown["samples"]

        for sample_idx, (p, sample) in enumerate(zip(task_parsed, samples)):
            if sample_idx >= k:
                # skip extra samples beyond k: later tasks may have more samples than the first
                continue
            if p.has_reasoning_block:
                block_counts[sample_idx] += 1

            if bool(p.reasoning.strip()):
                any_reasoning_counts[sample_idx] += 1

            if p.has_valid_reasoning:
                valid_reasoning_counts[sample_idx] += 1
                # conditional accuracy: only tasks where this sample has valid reasoning
                total_reasoning_counts[sample_idx] += 1
                if _is_correct(sample):
                    correct_reasoning_counts[sample_idx] += 1

            if two_pass:
                # reasoning_truncation: pass 1 hit the max_think_tokens cap
                if (
                    truncated_flags is not None
                    and truncated_flags[task_idx][sample_idx]
                ):
                    reasoning_truncation_counts[sample_idx] += 1
                # answer_overflow: pass 2 hit the token limit
                if p.is_overflow:
                    answer_overflow_counts[sample_idx] += 1
            else:
                # baseline: any overflow is flagged by truncated_flags (finish_reason=="length");
                # has_truncated_reasoning tells us if the cut happened inside the reasoning block
                overflowed = (
                    truncated_flags is not None
                    and truncated_flags[task_idx][sample_idx]
                )
                if overflowed and p.has_truncated_reasoning:
                    reasoning_overflow_counts[sample_idx] += 1
                if overflowed and not p.has_truncated_reasoning:
                    answer_overflow_counts[sample_idx] += 1

            # default True: old breakdown files without the field are treated as extracted
            if sample.get("answer_extracted", True):
                extracted_counts[sample_idx] += 1

            # leak accuracy: correct in answer or reasoning block
            if _is_correct_anywhere(sample):
                correct_leak_counts[sample_idx] += 1

    # compute per-sample-index rates
    reasoning_block_list = [c / total_tasks for c in block_counts]
    any_reasoning_list = [c / total_tasks for c in any_reasoning_counts]
    valid_reasoning_list = [c / total_tasks for c in valid_reasoning_counts]
    reasoning_truncation_rate_list = [
        c / total_tasks for c in reasoning_truncation_counts
    ]
    reasoning_overflow_rate_list = [c / total_tasks for c in reasoning_overflow_counts]
    answer_overflow_rate_list = [c / total_tasks for c in answer_overflow_counts]
    answer_extracted_rate_list = [c / total_tasks for c in extracted_counts]
    reasoning_accuracy_list: list[float | None] = [
        (correct_reasoning_counts[i] / total_reasoning_counts[i])
        if total_reasoning_counts[i]
        else None
        for i in range(k)
    ]
    # leak denominator is always total_tasks (guaranteed >= 1 here)
    reasoning_leak_accuracy_list: list[float] = [
        correct_leak_counts[i] / total_tasks for i in range(k)
    ]

    # average the per-sample-index rates for the scalar summary keys
    reasoning_block = sum(reasoning_block_list) / k
    any_reasoning = sum(any_reasoning_list) / k
    valid_reasoning = sum(valid_reasoning_list) / k
    reasoning_truncation_rate = sum(reasoning_truncation_rate_list) / k
    reasoning_overflow_rate = sum(reasoning_overflow_rate_list) / k
    answer_overflow_rate = sum(answer_overflow_rate_list) / k
    answer_extracted_rate = sum(answer_extracted_rate_list) / k
    # for conditional accuracy: average only the non-None entries
    valid_reasoning_acc = [v for v in reasoning_accuracy_list if v is not None]
    reasoning_accuracy: float | None = (
        sum(valid_reasoning_acc) / len(valid_reasoning_acc)
        if valid_reasoning_acc
        else None
    )
    reasoning_leak_accuracy: float = sum(reasoning_leak_accuracy_list) / len(
        reasoning_leak_accuracy_list
    )

    fmt_acc = f"{reasoning_accuracy:.1%}" if reasoning_accuracy is not None else "n/a"
    log(
        f"    Reasoning: block={reasoning_block:.1%}, any={any_reasoning:.1%}, "
        f"valid={valid_reasoning:.1%}, truncation={reasoning_truncation_rate:.1%}, "
        f"reasoning_overflow={reasoning_overflow_rate:.1%}, "
        f"answer_overflow={answer_overflow_rate:.1%}, extracted={answer_extracted_rate:.1%}, "
        f"accuracy={fmt_acc}, leak={reasoning_leak_accuracy:.1%}",
    )

    return {
        "reasoning_block": reasoning_block,
        "reasoning_block_list": reasoning_block_list,
        "any_reasoning": any_reasoning,
        "any_reasoning_list": any_reasoning_list,
        "valid_reasoning": valid_reasoning,
        "valid_reasoning_list": valid_reasoning_list,
        "reasoning_truncation_rate": reasoning_truncation_rate,
        "reasoning_truncation_rate_list": reasoning_truncation_rate_list,
        "reasoning_overflow_rate": reasoning_overflow_rate,
        "reasoning_overflow_rate_list": reasoning_overflow_rate_list,
        "answer_overflow_rate": answer_overflow_rate,
        "answer_overflow_rate_list": answer_overflow_rate_list,
        "answer_extracted_rate": answer_extracted_rate,
        "answer_extracted_rate_list": answer_extracted_rate_list,
        "reasoning_accuracy": reasoning_accuracy,
        "reasoning_accuracy_list": reasoning_accuracy_list,
        "reasoning_leak_accuracy": reasoning_leak_accuracy,
        "reasoning_leak_accuracy_list": reasoning_leak_accuracy_list,
    }


def aggregate_baseline_token_counts(token_counts: list[list[dict]]) -> dict:
    """Compute mean token statistics across all tasks and samples for baseline runs.

    Flattens the [task][sample] token count dicts (each with prompt_tokens and
    completion_tokens keys) and returns mean values for each field.

    Returns a dict of mean token stats, or {} if token_counts is empty.
    """
    flat = [sample for task in token_counts for sample in task]
    if not flat:
        return {}

    def mean(key: str) -> float:
        return sum(s[key] for s in flat) / len(flat)

    return {
        "mean_prompt_tokens": mean("prompt_tokens"),
        "mean_completion_tokens": mean("completion_tokens"),
    }


def aggregate_token_counts(token_counts: list[list[dict]]) -> dict:
    """Compute mean token statistics across all tasks and samples for two-pass runs.

    Flattens the [task][sample] token count dicts (each with pass1_prompt,
    pass1_reasoning, pass2_prompt, pass2_answer, and total_generated keys)
    and returns mean values for each field.

    Returns a dict of mean token stats, or {} if token_counts is empty.
    """
    flat = [sample for task in token_counts for sample in task]
    if not flat:
        return {}

    def mean(key: str) -> float:
        return sum(s[key] for s in flat) / len(flat)

    return {
        "mean_pass1_prompt": mean("pass1_prompt"),
        "mean_pass1_reasoning": mean("pass1_reasoning"),
        "mean_pass2_prompt": mean("pass2_prompt"),
        "mean_pass2_answer": mean("pass2_answer"),
        "mean_total_generated": mean("total_generated"),
    }
