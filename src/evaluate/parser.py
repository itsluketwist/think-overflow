"""Parsing of model responses into reasoning and answer components."""

import re
from dataclasses import dataclass


# tag patterns — capture group(1) holds the tag name (e.g. "think", "thinking")
_REASONING_CLOSE_TAG = re.compile(
    r"</(think|thinking|reasoning|thought)>",
    re.IGNORECASE,
)
_REASONING_OPEN_TAG = re.compile(
    r"<(think|thinking|reasoning|thought)>",
    re.IGNORECASE,
)


@dataclass
class ParsedResponse:
    """A model response split into reasoning and answer components."""

    answer: str  # text after the closing reasoning tag (or full response if none)
    reasoning: str  # raw content of the reasoning block (empty string if no reasoning)
    reasoning_tag: str | None  # tag name used, e.g. "think" (None if no tag found)
    has_reasoning_block: (
        bool  # True if any block structure present (including blank/truncated)
    )
    has_valid_reasoning: (
        bool  # True if a completed, non-blank reasoning block is present
    )
    has_truncated_reasoning: (
        bool  # True if reasoning started but the closing tag never appeared
    )
    is_overflow: (
        bool  # True if response hit the max_tokens limit (finish_reason == "length")
    )


def parse_response(
    response: str,
    olmo_style: bool = False,
    finish_reason: str = "stop",
) -> ParsedResponse:
    """Parse a single model response into its reasoning and answer components.

    Handles four formats:
    - standard: <think>content</think>answer
    - olmo-style: content</think>answer  (opening tag injected by chat template)
    - truncated standard: <think>content...  (open tag, no close tag)
    - truncated olmo: content...  (no tags; only detectable with olmo_style=True)

    Pass finish_reason="length" to mark the response as an overflow.
    Reasoning content is stored raw (unstripped) for hybrid-decoding prefix reconstruction.
    """
    is_overflow = finish_reason == "length"
    close_match = _REASONING_CLOSE_TAG.search(response)

    if close_match:
        tag = close_match.group(1).lower()
        # strip any open tag before the close tag (olmo-style has no open tag in decoded output)
        before_close = response[: close_match.start()]
        thinking = _REASONING_OPEN_TAG.sub("", before_close, count=1)
        answer = response[close_match.end() :].strip()
        has_valid_reasoning = bool(thinking.strip())
        return ParsedResponse(
            answer=answer,
            reasoning=thinking,
            reasoning_tag=tag,
            has_reasoning_block=True,
            has_valid_reasoning=has_valid_reasoning,
            has_truncated_reasoning=False,
            is_overflow=is_overflow,
        )

    # no closing tag — check for a truncated reasoning block (open tag present)
    open_match = _REASONING_OPEN_TAG.search(response)
    if open_match:
        # model started reasoning but output was cut off before the close tag
        return ParsedResponse(
            answer="",
            reasoning=response[open_match.end() :],
            reasoning_tag=open_match.group(1).lower(),
            has_reasoning_block=True,
            has_valid_reasoning=False,
            has_truncated_reasoning=True,
            is_overflow=is_overflow,
        )

    if olmo_style:
        # for olmo-style models the open tag is injected by the chat template and never
        # appears in output — a missing close tag means truncation, not a plain response
        return ParsedResponse(
            answer="",
            reasoning=response,
            reasoning_tag=None,
            has_reasoning_block=True,
            has_valid_reasoning=False,
            has_truncated_reasoning=True,
            is_overflow=is_overflow,
        )

    # no reasoning tags at all — plain response
    return ParsedResponse(
        answer=response,
        reasoning="",
        reasoning_tag=None,
        has_reasoning_block=False,
        has_valid_reasoning=False,
        has_truncated_reasoning=False,
        is_overflow=is_overflow,
    )


def parse_responses(
    responses: list[list[str]],
    olmo_style: bool = False,
    finish_reasons: list[list[str]] | None = None,
) -> list[list[ParsedResponse]]:
    """Parse all model responses into structured ParsedResponse objects.

    Pass finish_reasons (shape [task][sample], values "stop"/"length") to populate
    is_overflow on each ParsedResponse. If None, all responses default to is_overflow=False.

    Returns a nested list matching the shape of the input.
    """
    result = []
    for task_idx, sample_responses in enumerate(responses):
        task_result = []
        for sample_idx, r in enumerate(sample_responses):
            # resolve finish reason for this specific sample if provided
            if finish_reasons is not None:
                finish_reason = finish_reasons[task_idx][sample_idx]
            else:
                finish_reason = "stop"
            task_result.append(
                parse_response(
                    r,
                    olmo_style=olmo_style,
                    finish_reason=finish_reason,
                )
            )
        result.append(task_result)
    return result
