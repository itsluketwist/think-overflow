"""Evaluation of math/reasoning responses using math_verify."""

import re
from typing import Any

from math_verify import ExprExtractionConfig, LatexExtractionConfig, parse, verify

from src.evaluate.parser import ParsedResponse
from src.evaluate.statistics import compute_pass_at_1, wilson_ci


# for responses: prioritise \boxed{} then fall back to plain expressions/numbers
_RESPONSE_CONFIGS = (
    LatexExtractionConfig(boxed_match_priority=0),
    ExprExtractionConfig(),
)

# for gold answers: standard latex + plain expression extraction (no boxed priority needed)
_GOLD_CONFIGS = (
    LatexExtractionConfig(),
    ExprExtractionConfig(),
)

# matches gsm8k-style answers like "#### 310" or "####310"
_GSM8K_PATTERN = re.compile(r"####\s*(-?[\d,]+(?:\.\d+)?)")

# max length for raw string fallback (only short gold answers that fail parsing)
_MAX_FALLBACK_LEN = 100


def _parse_answer(text: str, configs: tuple) -> Any:
    """Parse a math answer string using math_verify; falls back to raw string for short text.

    Returns the parsed result list, or None if all extraction fails.
    """
    # guard against empty input before calling math_verify
    if not text.strip():
        return None

    try:
        result = parse(text, extraction_config=configs)
        if result:
            return result
    except Exception:
        pass

    # raw string fallback for short answers (e.g. plain integer gold values)
    if len(text.strip()) <= _MAX_FALLBACK_LEN:
        return [text.strip()]

    return None


def _extract_from_text(text: str, gsm8k_check: bool = True) -> Any:
    """Extract a math answer from raw text via gsm8k pre-extraction then math_verify.

    Returns the parsed result list, or None if extraction fails.
    """
    if not text.strip():
        return None

    # pre-extract gsm8k-style "#### 310" answers before general parsing,
    # since math_verify doesn't recognise this format natively; use the
    # last occurrence in case intermediate steps use the same format
    if gsm8k_check:
        gsm8k_matches = _GSM8K_PATTERN.findall(text)
        if gsm8k_matches:
            number_text = gsm8k_matches[-1].replace(",", "")
            return _parse_answer(number_text, configs=(ExprExtractionConfig(),))

    return _parse_answer(text, configs=_RESPONSE_CONFIGS)


def evaluate_math(
    responses: list[list[ParsedResponse]],
    records: list[dict[str, Any]],
    dataset_name: str | None = None,
) -> tuple[dict, list[dict]]:
    """Evaluate math responses by extracting \\boxed{}/latex/plain answers and verifying them.

    Evaluates the answer section only; also checks correct_in_reasoning as a diagnostic.

    Returns a dict with pass_at_1, pass_at_1_list, pass_at_k, k, total, and a per-task breakdown.
    """
    results = []
    k = len(responses[0]) if responses and responses[0] else 1
    # track correct count per sample index for averaged pass@1 calculation
    per_sample_correct = [0] * k
    pass_at_k_count = 0

    for sample_parsed, record in zip(responses, records):
        expected = record["answer"]

        # parse the gold answer with standard configs + raw fallback
        gold = _parse_answer(expected, configs=_GOLD_CONFIGS)

        # check each sampled response for this prompt
        sample_results = []
        for p in sample_parsed:
            # primary evaluation: answer section only
            # empty answer section = model failed to produce an answer after reasoning
            extracted = _extract_from_text(p.answer) if p.answer.strip() else None

            # verify the extracted answer against gold
            try:
                is_correct = verify(gold, extracted) if extracted else False
            except Exception:
                is_correct = False

            # diagnostic: check if the correct answer appeared in the reasoning block
            # (this does not affect the primary score)
            correct_in_reasoning: bool | None = None
            if p.reasoning.strip():
                reasoning_extracted = _extract_from_text(p.reasoning)
                try:
                    correct_in_reasoning = (
                        verify(gold, reasoning_extracted)
                        if reasoning_extracted
                        else False
                    )
                except Exception:
                    correct_in_reasoning = False

            sample_results.append(
                {
                    "extracted": str(extracted) if extracted else None,
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
                "gold": str(gold),
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
