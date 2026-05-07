"""Evaluation of multiple choice reasoning responses."""

import re
from typing import Any

from thinkpack import ParsedResponse

from src.evaluate.statistics import compute_pass_at_1, wilson_ci


# patterns applied to the full (uppercased) response text
_FULL_TEXT_PATTERNS = [
    r"(?:ANSWER|CHOICE)\s*(?:IS|:)\s*([A-Z])\b",  # "answer is X" or "answer: X"
    r"\\BOXED\{([A-Z])\}",  # \boxed{A}
    r"\b([A-Z])\s*[.\):]?\s*$",  # standalone letter at end
]

# patterns applied to only the last 200 characters (reduces false positives)
_TAIL_PATTERNS = [
    r"\b([A-Z])[.:\)\]]\s+\S+.*$",  # "D. R-loops" or "D: R-loops"
    r"[\(\[]([A-Z])[\)\]]",  # "(X)" or "[X]"
]


def _extract_choice(response: str) -> str | None:
    """Extract the answer letter (A, B, C, D, …) from a response using ordered patterns.

    Returns the letter if found, None otherwise.
    """
    text = response.upper()
    tail = text[-200:] if len(text) > 200 else text

    for pattern in _FULL_TEXT_PATTERNS:
        if match := re.search(pattern, text):
            return match.group(1)

    for pattern in _TAIL_PATTERNS:
        if match := re.search(pattern, tail):
            return match.group(1)

    return None


def evaluate_reasoning(
    responses: list[list[ParsedResponse]],
    records: list[dict[str, Any]],
    dataset_name: str | None = None,
) -> tuple[dict, list[dict]]:
    """Evaluate multiple-choice responses by extracting and comparing letter choices.

    Evaluates the answer section only; also checks correct_in_reasoning as a diagnostic.

    Returns a dict with pass_at_1, pass_at_1_list, pass_at_k, k, total, and a per-task breakdown.
    """
    results = []
    k = len(responses[0]) if responses and responses[0] else 1
    # track correct count per sample index for averaged pass@1 calculation
    per_sample_correct = [0] * k
    pass_at_k_count = 0

    for sample_parsed, record in zip(responses, records):
        expected = record["answer"].strip().upper()

        sample_results = []
        for p in sample_parsed:
            # primary evaluation: extract choice from answer section only
            # empty answer section = model failed to produce an answer after reasoning
            extracted = _extract_choice(p.answer) if p.answer.strip() else None
            is_correct = extracted == expected if extracted else False

            # diagnostic: check if the correct choice appeared in the reasoning block
            correct_in_reasoning: bool | None = None
            if p.reasoning.strip():
                reasoning_extracted = _extract_choice(p.reasoning)
                correct_in_reasoning = (
                    reasoning_extracted == expected if reasoning_extracted else False
                )

            sample_results.append(
                {
                    "extracted": extracted,
                    "answer_extracted": extracted is not None,
                    "correct": is_correct,
                    "correct_in_reasoning": correct_in_reasoning,
                }
            )

        # track correct per sample index — used to compute averaged pass@1 below
        # skip indices beyond k: later tasks may have more samples than the first
        for i, r in enumerate(sample_results):
            if i >= k:
                break
            if r["correct"]:
                per_sample_correct[i] += 1

        # pass@k: any sample correct
        any_correct = any(r["correct"] for r in sample_results)
        if any_correct:
            pass_at_k_count += 1

        # per-task pass@1 stored as first-sample bool (used in breakdown only)
        first_correct = sample_results[0]["correct"] if sample_results else False
        results.append(
            {
                "expected": expected,
                "samples": sample_results,
                "pass_at_1": first_correct,
                "pass_at_k": any_correct,
            }
        )

    total = len(results)
    # averaged pass@1: treat each sample index as an independent trial and average
    pass_at_1_list, pass_at_1, pass_at_1_ci = compute_pass_at_1(
        counts=per_sample_correct,
        total=total,
        k=k,
    )
    pass_at_k_ci = wilson_ci(count=pass_at_k_count, total=total)

    return {
        "pass_at_1": pass_at_1,
        "pass_at_1_ci": pass_at_1_ci,
        "pass_at_1_list": pass_at_1_list,
        "pass_at_k": pass_at_k_count / total if total else 0.0,
        "pass_at_k_ci": pass_at_k_ci,
        "k": k,
        "total": total,
    }, results
