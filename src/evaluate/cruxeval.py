"""Evaluation of CRUXEval responses (code understanding: input/output prediction)."""

import ast
import re
from typing import Any

from thinkpack import ParsedResponse

from src.evaluate.code import _execute_code
from src.evaluate.statistics import compute_pass_at_1, wilson_ci


# matches the [ANSWER]...[/ANSWER] block used in the original CRUXEval prompts
_ANSWER_BLOCK = re.compile(r"\[ANSWER\](.*?)\[/ANSWER\]", re.DOTALL | re.IGNORECASE)

# matches a markdown code fence (```python or ```) wrapping a block
_CODE_FENCE = re.compile(r"```(?:python|py)?\s*\n?(.*?)```", re.DOTALL | re.IGNORECASE)


def _extract_literal(text: str, task: str) -> str:
    """Extract a Python literal from a model response using a fallback chain.

    Tries: (1) last [ANSWER]...[/ANSWER] block, (1b) unclosed [ANSWER] tag,
    (2) == split (assertion style), (3) markdown fence, (4) last non-empty line.

    Returns the extracted candidate string (not yet validated as a literal).
    """
    stripped = text.strip()

    # 1. [ANSWER] block — primary format for original-paper-aligned prompts.
    #    anchor to the LAST [ANSWER] tag: models sometimes echo the few-shot example
    #    answers before their own, so the earliest match is often wrong.
    #    if the last [ANSWER] tag has a matching [/ANSWER], extract between them;
    #    otherwise (unclosed), take whatever text follows the tag.
    last_open = stripped.rfind("[ANSWER]")
    if last_open != -1:
        after_tag = stripped[last_open + len("[ANSWER]") :].strip()
        close_idx = after_tag.upper().find("[/ANSWER]")
        if close_idx != -1:
            # complete block — recurse on the inner content
            inner = after_tag[:close_idx].strip()
            if inner:
                return _extract_literal(text=inner, task=task)
        elif after_tag:
            # unclosed — take the first non-empty line after the tag
            lines = [ln.strip() for ln in after_tag.splitlines() if ln.strip()]
            if lines:
                return _extract_literal(text=lines[0], task=task)

    # 2. assertion pattern — mirrors the original CRUXEval extraction approach.
    #    the prompt ends with "assert f(...) == ??" so models often echo this form.
    if "==" in stripped:
        parts = stripped.split("==")
        if task == "output":
            # take everything after the last '==' (the predicted return value),
            # stripping any trailing markdown fence or whitespace
            candidate = parts[-1].strip().rstrip("`").strip()
        else:
            # take everything after 'assert f(' and before '==' (the predicted input)
            lhs = parts[0].strip()
            # strip "assert f(" prefix if present, leaving just the argument(s)
            if lhs.startswith("assert f("):
                lhs = lhs[len("assert f(") :].rstrip(")")
            candidate = lhs.strip()
        if candidate:
            return candidate

    # 3. markdown fence — strip ```...``` and return inner content
    fence_match = _CODE_FENCE.search(stripped)
    if fence_match:
        candidate = fence_match.group(1).strip()
        if candidate:
            return candidate

    # 4. last non-empty line — handles "The answer is: X" style responses
    lines = [ln.strip() for ln in stripped.splitlines() if ln.strip()]
    if lines:
        return lines[-1]

    return stripped


def _check_output_prediction(predicted: str, expected: str, task: str) -> bool:
    """Check if a predicted output matches expected via ast.literal_eval equality.

    Returns False for any parse or comparison error.
    """
    candidate = _extract_literal(text=predicted, task=task)
    try:
        return ast.literal_eval(candidate) == ast.literal_eval(expected.strip())
    except Exception:
        return False


def _check_input_prediction(predicted: str, code: str, expected_output: str) -> bool:
    """Check if a predicted input produces the expected output when run via f(predicted).

    Accepts any input that produces the correct output, not just the ground-truth input.

    Returns True if execution succeeds (assertion passes).
    """
    candidate = _extract_literal(text=predicted, task="input")
    # assemble: function definition + assertion that f(predicted) == expected
    test_code = f"{code}\n\nassert f({candidate}) == {expected_output}"
    success, _ = _execute_code(code=test_code)
    return success


def evaluate_cruxeval(
    responses: list[list[ParsedResponse]],
    records: list[dict[str, Any]],
    dataset_name: str | None = None,
) -> tuple[dict, list[dict]]:
    """Evaluate CRUXEval input/output prediction responses.

    Dispatches on record['task']: 'output' checks via ast.literal_eval; 'input' runs f(predicted).
    Evaluates the answer section only; also checks correct_in_reasoning as a diagnostic.

    Returns a dict with pass_at_1, pass_at_1_list, pass_at_k, k, total, and a per-task breakdown.
    """
    results = []
    k = len(responses[0]) if responses and responses[0] else 1
    # track correct count per sample index for averaged pass@1 calculation
    per_sample_correct = [0] * k
    pass_at_k_count = 0

    for sample_parsed, record in zip(responses, records):
        task = record["task"]  # "output" or "input"
        expected = record["answer"]

        sample_results = []
        for p in sample_parsed:
            raw = p.answer.strip()

            # primary evaluation: answer section only
            # empty answer = model failed to produce a prediction
            if raw:
                if task == "output":
                    correct = _check_output_prediction(
                        predicted=raw,
                        expected=expected,
                        task=task,
                    )
                else:
                    correct = _check_input_prediction(
                        predicted=raw,
                        code=record["code"],
                        expected_output=expected,
                    )
            else:
                correct = False

            # diagnostic: check if the correct answer appeared in the reasoning block
            correct_in_reasoning: bool | None = None
            if p.reasoning.strip():
                reasoning_raw = p.reasoning.strip()
                if task == "output":
                    correct_in_reasoning = _check_output_prediction(
                        predicted=reasoning_raw,
                        expected=expected,
                        task=task,
                    )
                else:
                    correct_in_reasoning = _check_input_prediction(
                        predicted=reasoning_raw,
                        code=record["code"],
                        expected_output=expected,
                    )

            sample_results.append(
                {
                    "predicted": _extract_literal(text=raw, task=task) if raw else None,
                    "answer_extracted": bool(raw),
                    "correct": correct,
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
                "task": task,
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
