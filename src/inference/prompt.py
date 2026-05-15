"""Prompt generation functions for vLLM inference."""

from __future__ import annotations

from typing import Any

import thinkpack
from vllm import LLM, SamplingParams


def _check_sampling_params(samples: int, temperature: float | None) -> None:
    """Raise if multiple samples are requested with greedy (zero temperature) decoding."""
    if samples > 1 and temperature == 0.0:
        raise ValueError(
            f"temperature must be > 0 when samples > 1 (got samples={samples}, "
            f"temperature={temperature}). Greedy decoding produces identical outputs.",
        )


def prompt_llm(
    llm: LLM,
    tokenizer: Any,
    prompts: list[str],
    samples: int,
    think_prefixes: list[str] | None = None,
    seed: int = 42,
    temperature: float | None = 0.0,
    top_p: float | None = 1.0,
    top_k: int | None = -1,
    max_tokens: int | list[int] | None = None,
    min_tokens: int = 32,
    stop: list[str] | None = None,
    chunk_size: int = 512,
) -> tuple[list[list[str]], list[list[str]], list[int], list[list[int]]]:
    """Prompt a vLLM model for responses.

    Uses thinkpack.apply_chat_template to build prompts in a model-agnostic way:
    - No think_prefixes (pass 1 / onepass): open tag is added to steer the model
      into its reasoning block. The model generates the reasoning.
    - With think_prefixes (pass 2): each prefix is injected as complete reasoning
      inside the closed think block; add_generation_reasoning=False ensures the
      model generates the answer directly after the closing tag.

    temperature/top_p/top_k accept None to use vLLM's defaults (1.0, 1.0, -1).
    max_tokens may be a list to set a per-prompt budget (e.g. pass 2 remaining budgets).

    Returns (responses, stop_reasons, prompt_tokens, completion_tokens).
    responses and stop_reasons are nested lists of shape [task][sample].
    stop_reason is "stop" (hit EOS/stop string) or "length" (hit max_tokens).
    prompt_tokens is shape [task]. completion_tokens is shape [task][sample].
    """
    _check_sampling_params(samples=samples, temperature=temperature)
    stop = stop or []

    # build chat-formatted prompt strings via thinkpack
    conversations = [[{"role": "user", "content": p}] for p in prompts]
    if think_prefixes is not None:
        # pass 2: inject reasoning then close the block so model generates only the answer.
        # response_prefix="" is what triggers thinkpack to add the close tag — without it the
        # block stays open and the model generates its own </think>, producing a duplicate.
        # (add_generation_reasoning=False cannot be combined with think_prefix; thinkpack errors)
        texts = [
            thinkpack.apply_chat_template(
                conversation=conv,
                tokenizer=tokenizer,
                think_prefix=tp,
                response_prefix="",
            )
            for conv, tp in zip(conversations, think_prefixes)
        ]
    else:
        # pass 1 / onepass: standard prompt with open tag appended
        texts = [
            thinkpack.apply_chat_template(
                conversation=conv,
                tokenizer=tokenizer,
            )
            for conv in conversations
        ]

    # resolve eos token for the stop list
    if hasattr(tokenizer, "eos_token") and tokenizer.eos_token is not None:
        eos_token_str = tokenizer.eos_token
    else:
        eos_token_str = tokenizer.decode([tokenizer.eos_token_id])

    # shared kwargs for building SamplingParams (all prompts use the same sampling settings)
    base_params: dict = {
        "seed": seed,
        "n": samples,
        "temperature": temperature if temperature is not None else 1.0,
        "top_p": top_p if top_p is not None else 1.0,
        "top_k": top_k if top_k is not None else -1,
        "min_tokens": min_tokens,
        "skip_special_tokens": True,
        "stop": [eos_token_str] + stop,
    }

    # when max_tokens is a list, each prompt gets its own SamplingParams with its own budget;
    # this is used in pass 2 to enforce the overall token cap per-sample
    per_prompt = isinstance(max_tokens, list)
    shared_params = SamplingParams(
        **base_params, max_tokens=None if per_prompt else max_tokens
    )

    # process in chunks to avoid the vllm 0.7.x async-stopped scheduler bug
    all_completions = []
    for start in range(0, len(texts), chunk_size):
        chunk_texts = texts[start : start + chunk_size]
        if per_prompt:
            assert isinstance(max_tokens, list)
            chunk_params: list[SamplingParams] = [
                SamplingParams(**base_params, max_tokens=mt)
                for mt in max_tokens[start : start + chunk_size]
            ]
        else:
            chunk_params = shared_params
        all_completions.extend(
            llm.generate(
                prompts=chunk_texts,
                sampling_params=chunk_params,
                use_tqdm=True,
            )
        )

    stop_reasons = [
        [output.finish_reason or "stop" for output in c.outputs]
        for c in all_completions
    ]
    prompt_tokens = [len(c.prompt_token_ids) for c in all_completions]
    completion_tokens = [
        [len(output.token_ids) for output in c.outputs] for c in all_completions
    ]
    responses = [[output.text for output in c.outputs] for c in all_completions]
    return responses, stop_reasons, prompt_tokens, completion_tokens
