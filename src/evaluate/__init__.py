"""Evaluation dispatch for different response types."""

from typing import Any, Callable

import thinkpack
from thinkpack import ParsedResponse

from src.evaluate.code import evaluate_code
from src.evaluate.cruxeval import evaluate_cruxeval
from src.evaluate.math import evaluate_math
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
    tokenizer: Any | None = None,
    details: list[list[dict]] | None = None,
    dataset_name: str | None = None,
) -> tuple[dict, list[dict]]:
    """Evaluate model responses against ground truth, dispatching on eval_type.

    eval_type options: math (\\boxed{} extraction), reasoning (letter choice), code (execution), crux.
    Pass tokenizer so thinkpack auto-detects the model's reasoning tag style.
    Pass details ([task][sample] dicts) for transition and overflow rate computation;
    the details structure is self-describing (transition=False for onepass samples).
    Pass dataset_name (file stem) so evaluators can apply dataset-specific behaviour
    (e.g. skipping test execution for benchmarks with official harnesses).

    Returns a result dict (including a 'statistics' sub-dict) and a per-task breakdown list.
    """
    if eval_type not in _EVALUATORS:
        supported = ", ".join(_EVALUATORS.keys())
        raise ValueError(
            f"unknown evaluation type '{eval_type}'. supported types: {supported}",
        )

    # parse responses — thinkpack handles nested list[list[str]] and detects tag style
    parsed: list[list[ParsedResponse]] = thinkpack.parse(
        response=responses,
        tokenizer=tokenizer,
    )

    # run the type-specific evaluator
    evaluator = _EVALUATORS[eval_type]
    result, breakdown = evaluator(
        responses=parsed,
        records=records,
        dataset_name=dataset_name,
    )

    # compute reasoning statistics, blending thinkpack reliability metrics with
    # our custom transition/overflow rates derived from the details structure
    result["statistics"] = compute_reasoning_statistics(
        parsed_responses=parsed,
        details=details,
        results_breakdown=breakdown,
    )

    return result, breakdown
