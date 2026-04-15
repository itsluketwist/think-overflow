"""Evaluation dispatch for different response types."""

from typing import Any, Callable

from src.evaluate.code import evaluate_code
from src.evaluate.cruxeval import evaluate_cruxeval
from src.evaluate.math import evaluate_math
from src.evaluate.parser import ParsedResponse, parse_responses
from src.evaluate.reasoning import evaluate_reasoning
from src.evaluate.statistics import compute_reasoning_statistics


# registry of evaluation functions by type
_EVALUATORS: dict[str, Callable[..., tuple[dict[str, Any], list[dict[str, Any]]]]] = {
    "math": evaluate_math,
    "reasoning": evaluate_reasoning,  # multiple choice (GPQA, MuSR)
    "code": evaluate_code,
    "crux": evaluate_cruxeval,  # code understanding: input/output prediction
}

# eval types that need the boxed answer instruction appended to prompts
BOXED_EVAL_TYPES = {"math"}


def evaluate(
    responses: list[list[str]],
    records: list[dict[str, Any]],
    eval_type: str,
    olmo_style: bool = False,
    finish_reasons: list[list[str]] | None = None,
    truncated_flags: list[list[bool]] | None = None,
    two_pass: bool = False,
) -> tuple[dict, list[dict]]:
    """Evaluate model responses against ground truth, dispatching on eval_type.

    eval_type options: math (\\boxed{} extraction), reasoning (letter choice), code (execution), crux.
    Set olmo_style=True for models whose chat template injects the opening reasoning tag.
    Pass finish_reasons ([task][sample], "stop"/"length") for accurate overflow tracking.
    Pass truncated_flags ([task][sample] booleans) for overflow rate computation —
    two-pass: pass 1 cap-hit flags; baseline: finish_reason=="length" flags.
    Set two_pass=True to use two-pass overflow logic (see compute_reasoning_statistics).

    Returns a result dict (including a 'statistics' sub-dict) and a per-task breakdown list.
    """
    if eval_type not in _EVALUATORS:
        supported = ", ".join(_EVALUATORS.keys())
        raise ValueError(
            f"unknown evaluation type '{eval_type}'. supported types: {supported}",
        )

    # parse each response once — splits reasoning from answer, computes reasoning flags;
    # finish_reasons is passed here so is_overflow is set on each ParsedResponse directly
    parsed: list[list[ParsedResponse]] = parse_responses(
        responses,
        olmo_style=olmo_style,
        finish_reasons=finish_reasons,
    )

    # run the type-specific evaluator — passes ParsedResponse objects so evaluators
    # can inspect both the answer section and the reasoning content separately
    evaluator = _EVALUATORS[eval_type]
    result, breakdown = evaluator(
        responses=parsed,
        records=records,
    )

    # compute reasoning statistics over all parsed responses
    result["statistics"] = compute_reasoning_statistics(
        parsed_responses=parsed,
        results_breakdown=breakdown,
        truncated_flags=truncated_flags,
        two_pass=two_pass,
    )

    return result, breakdown
