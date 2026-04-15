"""Prompt generation functions for vLLM inference."""

from __future__ import annotations

from vllm import LLM, SamplingParams


def apply_chat_template(
    llm: LLM,
    prompts: list[str],
) -> tuple[list[str], str]:
    """Apply chat template to prompts and resolve the EOS token.

    Wraps each prompt in a user message, applies the tokenizer's
    chat template, and finds the EOS token string for stop conditions.

    Returns (templated_texts, eos_token_str).
    """
    tokenizer = llm.get_tokenizer()

    # get the eos token string
    if hasattr(tokenizer, "eos_token") and tokenizer.eos_token is not None:
        eos_token_str = tokenizer.eos_token
    else:
        eos_token_str = tokenizer.decode([tokenizer.eos_token_id])

    texts = []
    for prompt in prompts:
        messages = [
            {"role": "user", "content": prompt},
        ]
        result = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        # handle case where tokenizer returns token ids despite `tokenize=False`
        if isinstance(result, list):
            result = tokenizer.decode(result)

        texts.append(result)

    return texts, eos_token_str


def _check_sampling_params(samples: int, temperature: float) -> None:
    """Raise if multiple samples are requested with greedy (zero temperature) decoding."""
    if samples > 1 and temperature == 0.0:
        raise ValueError(
            f"temperature must be > 0 when samples > 1 (got samples={samples}, "
            f"temperature={temperature}). Greedy decoding produces identical outputs.",
        )


def _build_prefixed_texts(
    templated_texts: list[str],
    prefixes: list[str],
) -> list[str]:
    """Append reasoning prefixes to chat-templated prompt strings.

    For models where the chat template already injects the opening reasoning tag
    (e.g. OLMo ends its generation prompt with '<think>'), the opening tag line
    is stripped from the prefix to avoid duplication. For other models (e.g.
    Qwen3) the full prefix is appended as-is.

    Each prefix is either "" (no reasoning) or a full "<tag>\\n{body}</{tag}>\\n"
    block as produced by load_reasoning_prefixes.

    Returns a list of prompt strings ready to pass to vLLM.
    """
    phase2_texts = []
    for text, prefix in zip(templated_texts, prefixes):
        if not prefix:
            phase2_texts.append(text)
        else:
            # the first line of the prefix is the opening tag, e.g. "<think>"
            open_tag, body_and_close = prefix.split("\n", 1)
            if text.rstrip("\n").endswith(open_tag):
                # chat template already appended the opening tag — skip it to avoid duplication
                phase2_texts.append(text + body_and_close)
            else:
                # chat template does not add the opening tag — include the full prefix
                phase2_texts.append(text + prefix)
    return phase2_texts


def prompt_llm(
    llm: LLM,
    prompts: list[str],
    samples: int,
    prefixes: list[str] | None = None,
    seed: int = 42,
    temperature: float = 0.0,
    top_p: float = 1.0,
    top_k: int = -1,
    max_tokens: int = 8192,
    min_tokens: int = 32,
    stop: list[str] | None = None,
    chunk_size: int = 512,
) -> tuple[list[list[str]], list[list[str]], list[int], list[list[int]]]:
    """Prompt a vLLM model for responses.

    If prefixes are provided, each is appended to the chat-templated prompt
    (handling whether the model's chat template already injects the opening tag)
    and then prepended to the stored response so evaluation code can detect
    reasoning. Empty prefixes produce plain responses.

    Returns (responses, finish_reasons, prompt_tokens, completion_tokens).
    responses and finish_reasons are nested lists of shape [task][sample].
    finish_reason is "stop" (hit EOS) or "length" (hit max_tokens).
    prompt_tokens is shape [task] (one count per task, identical across samples).
    completion_tokens is shape [task][sample].
    """
    _check_sampling_params(samples=samples, temperature=temperature)
    stop = stop or []
    texts, eos_token_str = apply_chat_template(llm=llm, prompts=prompts)

    if prefixes is not None:
        texts = _build_prefixed_texts(templated_texts=texts, prefixes=prefixes)

    sampling_params = SamplingParams(
        seed=seed,
        n=samples,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        max_tokens=max_tokens,
        min_tokens=min_tokens,
        skip_special_tokens=True,
        stop=[eos_token_str] + stop,
    )

    # process in chunks to avoid the vllm 0.7.x async-stopped scheduler bug
    # (assert len(self._async_stopped) == 0); only observed during distillation runs.
    # distill profiles use chunk_size=128, standard inference uses chunk_size=512.
    all_completions = []
    for start in range(0, len(texts), chunk_size):
        all_completions.extend(
            llm.generate(
                prompts=texts[start : start + chunk_size],
                sampling_params=sampling_params,
                use_tqdm=True,
            )
        )

    # extract finish reasons — "stop" means hit EOS, "length" means hit max_tokens
    finish_reasons = [
        [output.finish_reason or "stop" for output in c.outputs]
        for c in all_completions
    ]

    # extract token counts — prompt tokens are identical across samples for a task
    prompt_tokens = [len(c.prompt_token_ids) for c in all_completions]
    completion_tokens = [
        [len(output.token_ids) for output in c.outputs] for c in all_completions
    ]

    if prefixes is not None:
        # prepend the full reasoning block to every sample so the stored
        # response includes the <think>...</think> block for evaluation
        responses = [
            [prefix + output.text for output in c.outputs]
            for prefix, c in zip(prefixes, all_completions)
        ]
        return responses, finish_reasons, prompt_tokens, completion_tokens
    responses = [[output.text for output in c.outputs] for c in all_completions]
    return responses, finish_reasons, prompt_tokens, completion_tokens
