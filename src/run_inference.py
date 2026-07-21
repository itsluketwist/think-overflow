"""Two-pass capped-reasoning inference for the think-overflow method.

Standard thinking-mode models (Qwen3, OLMo-Think) may spend their entire token
budget on reasoning without ever producing an answer. This module caps the
reasoning phase at max_think_tokens, appends a recovery suffix to truncated
traces, then feeds the completed (or capped) thinking block back into the model
to generate an answer.
"""

import traceback
from pathlib import Path
from typing import Any

import thinkpack
from llm_cgr import load_json, save_json
from transformers import AutoTokenizer
from vllm import LLM

from src.evaluate import evaluate
from src.evaluate.statistics import aggregate_token_counts
from src.inference.data import (
    construct_inference_prompts,
    load_dataset_records,
    parse_dataset_entry,
)
from src.inference.load import load_llm
from src.inference.prompt import prompt_llm
from src.inference.results import json_results
from src.utils.config import load_yaml_config
from src.utils.log import log, log_header, log_step
from src.utils.timer import log_timer


# named overflow suffixes — appended to truncated reasoning before closing </think>.
# "base" adds nothing (pure truncation), the others add an explicit recovery signal.
_OVERFLOW_SUFFIXES: dict[str, str] = {
    "base": "",
    "dots": "...",
    "truncated": " [reasoning truncated]",
    "formal": "... I have to stop thinking and answer now.",
    "human": "... oops, I really need to stop thinking and to answer.",
}


# named prompt suffixes — appended to every user prompt (onepass only) to test whether
# prompt-engineering shortens reasoning. "none" is the baseline (nothing appended).
_PROMPT_SUFFIXES: dict[str, str] = {
    "none": "",
    "concise": (
        "\n\nKeep your reasoning brief and to the point, then make sure you stop "
        "reasoning and provide a complete final answer."
    ),
    "aspects": (
        "\n\nDo not reason step by step. Instead, briefly consider the important "
        "aspects of the problem — such as the inputs, outputs, edge cases, and any "
        "interesting design choices — then stop and give your final answer."
    ),
}


def _run_onepass(
    llm: LLM,
    tokenizer: Any,
    prompts: list[str],
    samples: int,
    seed: int,
    temperature: float | None,
    top_p: float | None,
    top_k: int | None,
    max_tokens: int | None,
    chunk_size: int,
) -> list[list[dict]]:
    """Run single-pass unconstrained inference over a list of prompts.

    Returns details: a [task][sample] list of dicts with keys:
    text, stop_reason, truncated, transition, prompt, prompt_tokens_1, completion_tokens_1,
    prompt_tokens_2, completion_tokens_2. Pass 2 keys are None for onepass.
    transition is always False for onepass (no forced cap).
    truncated is True when stop_reason is "length" (hit the token limit).
    """
    log("    Generating responses (onepass)...")
    with log_timer("generation"):
        results = prompt_llm(
            llm=llm,
            tokenizer=tokenizer,
            prompts=prompts,
            samples=samples,
            think_prefixes=None,
            seed=seed,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            max_tokens=max_tokens,
            min_tokens=1,
            stop=None,
            chunk_size=chunk_size,
        )

    details: list[list[dict]] = []
    for r in results:
        task_details: list[dict] = []
        for j in range(samples):
            stop_reason = r["stop_reasons"][j]
            task_details.append(
                {
                    "prompt": r["prompt"],
                    "text": r["responses"][j],
                    "stop_reason": stop_reason,
                    "truncated": stop_reason == "length",  # hit model token limit
                    "transition": False,  # no forced cap in onepass
                    "prompt_tokens_1": r["prompt_tokens"],
                    "completion_tokens_1": r["completion_tokens"][j],
                    "prompt_tokens_2": None,
                    "completion_tokens_2": None,
                },
            )
        details.append(task_details)

    return details


def _run_twopass(
    llm: LLM,
    prompts: list[str],
    tokenizer: Any,
    max_think_tokens: int,
    overflow_suffix: str,
    max_tokens: int | None,
    samples: int,
    seed: int,
    temperature: float | None,
    top_p: float | None,
    top_k: int | None,
    chunk_size: int,
) -> list[list[dict]]:
    """Run two-pass capped reasoning inference over a list of prompts.

    Pass 1 generates reasoning up to max_think_tokens (stopping at the close tag
    if the model closes its thinking naturally before the cap). Samples that hit
    the cap trigger a transition: overflow_suffix is appended and the reasoning
    block is closed before pass 2 generates the answer.

    Returns details: a [task][sample] list of dicts with keys:
    text, stop_reason, truncated, transition, prompt_1, prompt_2, prompt_tokens_1,
    completion_tokens_1, prompt_tokens_2, completion_tokens_2.
    transition=True means the reasoning was forcibly capped (pass 1 hit max_think_tokens).
    truncated=True means pass 2 answer generation hit the token limit.
    """
    n_prompts = len(prompts)
    # get model tag info once for stop condition and response reconstruction
    model_info = thinkpack.get_model_info(tokenizer)

    # --- pass 1: generate reasoning, stopping at close tag or max_think_tokens ---
    log("    Pass 1: generating capped reasoning...")
    with log_timer("pass 1"):
        p1 = prompt_llm(
            llm=llm,
            tokenizer=tokenizer,
            prompts=prompts,
            samples=samples,
            think_prefixes=None,
            seed=seed,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            max_tokens=max_think_tokens,
            min_tokens=1,
            stop=[model_info.close_tag],
            chunk_size=chunk_size,
        )

    # count how many reasoning traces were cut short and triggered a transition
    transition_count = sum(
        1 for r in p1 for reason in r["stop_reasons"] if reason == "length"
    )
    log(
        f"    Transition: {transition_count}/{n_prompts * samples} "
        f"({transition_count / max(n_prompts * samples, 1):.1%})",
    )

    # --- build pass 2 think_prefixes: one per (prompt, sample) pair ---
    # flattened in row-major order: all samples for prompt 0, then prompt 1, ...
    # think_prefix is the raw reasoning content; thinkpack injects it as a completed
    # <think>...</think> block in the pass 2 prompt via apply_chat_template
    all_think_prefixes_flat: list[str] = []
    all_transition_flat: list[bool] = []

    for r in p1:
        for response, reason in zip(r["responses"], r["stop_reasons"]):
            transition = bool(reason == "length")
            think_prefix = response.strip() + (overflow_suffix if transition else "")
            all_think_prefixes_flat.append(think_prefix)
            all_transition_flat.append(transition)

    # expand prompts to match flattened (prompt × sample) pairs
    flat_prompts = [p for p in prompts for _ in range(samples)]

    # compute per-sample remaining budget for pass 2: overall cap minus reasoning tokens used
    # if pass 1 hit the length stop, use max_think_tokens (the cap) rather than the actual count
    # — keeps budget accounting consistent with what was intended even if vllm fell slightly short
    p2_max_tokens: list[int] | None
    if max_tokens is not None:
        p2_max_tokens = [
            max(
                1,
                max_tokens
                - (
                    max_think_tokens
                    if p1[i]["stop_reasons"][j] == "length"
                    else p1[i]["completion_tokens"][j]
                ),
            )
            for i in range(n_prompts)
            for j in range(samples)
        ]
    else:
        p2_max_tokens = None

    # --- pass 2: generate answers given the completed reasoning block ---
    log("    Pass 2: generating answers from reasoning...")
    with log_timer("pass 2"):
        p2 = prompt_llm(
            llm=llm,
            tokenizer=tokenizer,
            prompts=flat_prompts,
            samples=1,
            think_prefixes=all_think_prefixes_flat,
            seed=seed,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            max_tokens=p2_max_tokens,
            min_tokens=1,
            stop=None,
            chunk_size=chunk_size,
        )

    # --- reshape results from flat [n_prompts*samples] back to [task][sample] ---
    details: list[list[dict]] = []

    for i in range(n_prompts):
        task_details: list[dict] = []

        for j in range(samples):
            # flat index for this (prompt i, sample j) pair
            k = i * samples + j
            stop_reason = p2[k]["stop_reasons"][0]

            # split prompt_2 at the last open tag and append the answer as-is.
            p2_response = p2[k]["responses"][0]
            regenerated_close_tag = p2_response.strip().startswith(model_info.close_tag)
            prompt_2_text = p2[k]["prompt"]
            last_open = prompt_2_text.rfind(model_info.open_tag)
            full_text = prompt_2_text[last_open:] + p2_response

            task_details.append(
                {
                    "prompt_1": p1[i]["prompt"],
                    "prompt_2": p2[k]["prompt"],
                    "text": full_text,
                    "stop_reason": stop_reason,
                    "truncated": stop_reason == "length",  # answer generation hit limit
                    "transition": all_transition_flat[
                        k
                    ],  # reasoning was forcibly capped
                    "regenerated_close_tag": regenerated_close_tag,  # model re-emitted </think>
                    "prompt_tokens_1": p1[i]["prompt_tokens"],
                    "completion_tokens_1": p1[i]["completion_tokens"][j],
                    "prompt_tokens_2": p2[k]["prompt_tokens"],
                    "completion_tokens_2": p2[k]["completion_tokens"][0],
                },
            )

        details.append(task_details)

    return details


def _run_nothink(
    llm: LLM,
    tokenizer: Any,
    prompts: list[str],
    samples: int,
    seed: int,
    temperature: float | None,
    top_p: float | None,
    top_k: int | None,
    max_tokens: int | None,
    chunk_size: int,
) -> list[list[dict]]:
    """Run nothink inference: inject an empty <think></think> block then generate the answer.

    No reasoning pass is performed. Each prompt receives an empty think_prefix so
    thinkpack closes the block immediately, forcing the model to answer without reasoning.

    Returns details: a [task][sample] list of dicts with keys:
    text, stop_reason, truncated, transition, prompt, prompt_tokens_1, completion_tokens_1,
    prompt_tokens_2, completion_tokens_2. Pass 1 keys are None (no reasoning pass).
    transition is always False. regenerated_close_tag flags if the model re-emitted </think>.
    """
    model_info = thinkpack.get_model_info(tokenizer)

    # empty string as think_prefix causes thinkpack to produce <think></think>,
    # preventing the model from reasoning at all before answering
    empty_prefixes = ["" for _ in prompts]

    log("    Generating answers with empty think block (nothink)...")
    with log_timer("generation"):
        results = prompt_llm(
            llm=llm,
            tokenizer=tokenizer,
            prompts=prompts,
            samples=samples,
            think_prefixes=empty_prefixes,
            seed=seed,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            max_tokens=max_tokens,
            min_tokens=1,
            stop=None,
            chunk_size=chunk_size,
        )

    details: list[list[dict]] = []
    for r in results:
        task_details: list[dict] = []
        for j in range(samples):
            stop_reason = r["stop_reasons"][j]
            response = r["responses"][j]

            # reconstruct full text from the injected <think></think> block onward,
            # same approach as twopass: find the last open tag in the prompt
            prompt_text = r["prompt"]
            regenerated_close_tag = response.strip().startswith(model_info.close_tag)
            last_open = prompt_text.rfind(model_info.open_tag)
            full_text = prompt_text[last_open:] + response

            task_details.append(
                {
                    "prompt": r["prompt"],
                    "text": full_text,
                    "stop_reason": stop_reason,
                    "truncated": stop_reason == "length",
                    "transition": False,  # no reasoning cap — no transition possible
                    "regenerated_close_tag": regenerated_close_tag,
                    "prompt_tokens_1": None,
                    "completion_tokens_1": None,
                    "prompt_tokens_2": r["prompt_tokens"],
                    "completion_tokens_2": r["completion_tokens"][j],
                },
            )
        details.append(task_details)

    return details


def run_inference(
    model: str,
    model_config: str,
    config_file: str,
    config_profile: str,
    output: str,
    datasets: str,
    eval_type: str | None,
    max_think_tokens: int | None,
    max_tokens: int | None,
    overflow_suffix: str,
    prompt_suffix: str,
    update: bool,
    debug: bool = False,
) -> None:
    """Run two-pass, onepass, or nothink inference with explicit configuration arguments.

    The overflow_suffix argument is a key from _OVERFLOW_SUFFIXES (e.g. 'formal'),
    not the resolved string. When max_think_tokens is None, runs onepass (unconstrained).
    When max_think_tokens is 0, runs nothink (empty think block, no reasoning pass).
    The prompt_suffix argument is a key from _PROMPT_SUFFIXES; it appends an instruction
    to every user prompt and is only valid in onepass mode.
    """
    # resolve suffix key to the actual string appended to truncated reasoning
    overflow_suffix_str: str = _OVERFLOW_SUFFIXES[overflow_suffix]

    # resolve prompt suffix key to the actual string appended to every user prompt
    prompt_suffix_str: str = _PROMPT_SUFFIXES[prompt_suffix]

    # prompt suffixes are an onepass-only experiment — reject other modes explicitly
    if prompt_suffix != "none" and max_think_tokens is not None:
        raise ValueError(
            f"prompt_suffix ({prompt_suffix!r}) is only supported in onepass mode "
            f"(omit --max-think-tokens); got max_think_tokens={max_think_tokens}.",
        )

    # run name encodes the mode and token budgets for unique, readable output filenames.
    # output_subdir separates results by mode; results_suffix names the summary file.
    if max_think_tokens is None:
        # append the prompt-suffix key only when set so the baseline filename is unchanged
        suffix_tag = "" if prompt_suffix == "none" else f"_{prompt_suffix}"
        run_name = f"mx{max_tokens}_onepass{suffix_tag}"
        output_subdir = "onepass"
        results_suffix = "_onepass"
    elif max_think_tokens == 0:
        run_name = f"mx{max_tokens}_nothink"
        output_subdir = "nothink"
        results_suffix = "_nothink"
    else:
        run_name = f"mx{max_tokens}_th{max_think_tokens}_{overflow_suffix}"
        output_subdir = "twopass"
        results_suffix = "_twopass"

    if max_think_tokens == 0:
        mode_str = "nothink (empty think block)"
    elif max_think_tokens is None:
        mode_str = "onepass (unconstrained)"
    else:
        mode_str = "two-pass overflow"

    log_header("INFERENCE")
    log(f"Model: {model}")
    log(f"Config: {config_file} [{config_profile}]")
    log(f"Mode: {mode_str}")
    if max_think_tokens is not None and max_think_tokens > 0:
        log(f"Max think tokens: {max_think_tokens}")
        log(f"Overflow suffix: {overflow_suffix} ({repr(overflow_suffix_str)})")
    if max_think_tokens is None:
        log(f"Prompt suffix: {prompt_suffix} ({repr(prompt_suffix_str)})")
    log(f"Run name: {run_name}")
    log(f"Update: {update}")

    # load inference generation parameters from the selected profile
    inference_config = load_yaml_config(
        config_file=config_file,
        key=config_profile,
    )

    # load the model config from models.yaml
    model_cfg = load_yaml_config(
        config_file=model_config,
        key=model,
    )

    # validate token budgets: thinking cap must leave room for the answer
    if max_think_tokens is not None and max_tokens is not None:
        if max_think_tokens >= max_tokens:
            raise ValueError(
                f"max_think_tokens ({max_think_tokens}) must be less than "
                f"max_tokens ({max_tokens}) — no budget left for the answer.",
            )
        headroom = max_tokens - max_think_tokens
        if headroom < 512:
            log(
                f"WARNING: only {headroom} tokens remain for the answer after "
                f"the reasoning cap ({max_think_tokens}/{max_tokens}) — answers may be truncated.",
            )

    samples: int = inference_config.get("samples", 1)

    # sampling param resolution: inference config takes priority; when null (e.g. "default"
    # profile), fall back to per-model defaults from models.yaml; None if neither is set
    # (prompt_llm then uses vLLM's own defaults: temperature=1.0, top_p=1.0, top_k=-1)
    model_defaults = model_cfg.get("defaults", {})
    temperature: float | None = inference_config.get("temperature")
    if temperature is None:
        temperature = model_defaults.get("temperature")
    top_p: float | None = inference_config.get("top_p")
    if top_p is None:
        top_p = model_defaults.get("top_p")
    top_k: int | None = inference_config.get("top_k")
    if top_k is None:
        top_k = model_defaults.get("top_k")

    log(f"Max tokens (pass 2 answer budget): {max_tokens}")
    log(f"Samples: {samples}")
    log(f"Temperature: {temperature if temperature is not None else 'default (1.0)'}")
    log(f"Top-p: {top_p if top_p is not None else 'default (1.0)'}")
    log(f"Top-k: {top_k if top_k is not None else 'default (-1)'}")

    if debug:
        # debug runs go to a fixed directory so they never pollute real results
        output_dir = Path(output) / "debug" / model
    else:
        # onepass, twopass, and nothink results are kept in separate directories
        output_dir = Path(output) / output_subdir / model
    output_dir.mkdir(parents=True, exist_ok=True)
    log(f"Output directory: {output_dir}")

    # parse all dataset entries and determine which need generation
    dataset_entries = [parse_dataset_entry(d) for d in datasets.split(",") if d]

    if debug:
        # keep only the first dataset of each eval type for a quick smoke test
        seen_types: set[str | None] = set()
        debug_entries = []
        for entry in dataset_entries:
            t = entry[1]
            if t not in seen_types:
                seen_types.add(t)
                debug_entries.append(entry)
        dataset_entries = debug_entries
        log("Debug mode: one dataset per type, 5 samples each")
    all_datasets: list[tuple[str, Path, str | None]] = []
    datasets_to_generate: list[tuple[str, Path, str | None]] = []

    log()
    log(f"Datasets ({len(dataset_entries)}):")
    for dataset_path, dataset_eval_type in dataset_entries:
        dataset_stem = Path(dataset_path).stem
        file_path = output_dir / f"{dataset_stem}_{config_profile}_{run_name}.json"

        # fall back to the eval_type argument if auto-detection didn't find a type
        resolved_eval_type = dataset_eval_type or eval_type
        all_datasets.append((dataset_path, file_path, resolved_eval_type))

        needs_generation = update or not file_path.exists()
        if needs_generation:
            datasets_to_generate.append((dataset_path, file_path, resolved_eval_type))

        status = "generate" if needs_generation else "exists"
        log(f"  - {dataset_path} (type={resolved_eval_type}, {status})")

    # tokenizer is obtained from the loaded LLM during phase 1; if phase 1 is
    # skipped it is loaded via AutoTokenizer for use in evaluation
    tokenizer = AutoTokenizer.from_pretrained(
        pretrained_model_name_or_path=model_cfg["model_path"],
        trust_remote_code=True,
    )

    # --- phase 1: generation ---
    if datasets_to_generate:
        log_header(f"PHASE 1: GENERATION ({len(datasets_to_generate)} datasets)")

        log(f"Loading model: {model_cfg['model_path']}")
        llm = load_llm(
            model_path=model_cfg["model_path"],
            max_model_len=model_cfg.get("max_model_len"),
            gpu_memory_utilization=inference_config.get("gpu_memory_utilization"),
            max_num_seqs=inference_config.get("max_num_seqs"),
        )
        log("Model loaded successfully.")
        mi = thinkpack.get_model_info(tokenizer=tokenizer)
        log(f"Model info: prefixed={mi.prefixed}, tag={mi.tag_content}")
        log()

        for i, (dataset_path, file_path, resolved_eval_type) in enumerate(
            datasets_to_generate,
            start=1,
        ):
            log_step(i, len(datasets_to_generate), f"{dataset_path}")
            log(f"    Eval type: {resolved_eval_type}")

            dataset = load_dataset_records(
                dataset=dataset_path,
                default_dir="data",
                split="test",
                required_keys={"prompt"},
            )
            if debug:
                dataset = dataset[:5]
            log(f"    Loaded {len(dataset)} prompts")

            prompts = construct_inference_prompts(
                records=dataset,
                eval_type=resolved_eval_type,
                # prompt suffix is onepass-only; None for twopass/nothink modes
                prompt_suffix=prompt_suffix_str if max_think_tokens is None else None,
            )

            output_data: dict[str, Any]
            if max_think_tokens is None:
                details = _run_onepass(
                    llm=llm,
                    tokenizer=tokenizer,
                    prompts=prompts,
                    samples=samples,
                    seed=inference_config.get("seed", 42),
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                    max_tokens=max_tokens,
                    chunk_size=inference_config.get("chunk_size", 512),
                )
                output_data = {
                    "metadata": {
                        "model": model,
                        "dataset": dataset_path,
                        "config_profile": config_profile,
                        "config": inference_config,
                        "eval_type": resolved_eval_type,
                        "samples": samples,
                        "max_tokens": max_tokens,
                        "run_name": run_name,
                        "onepass": True,
                        "prompt_suffix_key": prompt_suffix,
                        "prompt_suffix": prompt_suffix_str,
                    },
                    "analysis": None,
                    "evaluation": None,
                    "details": details,
                }
                save_json(data=output_data, file_path=str(file_path))
                log(f"    Saved to {file_path}")

            elif max_think_tokens == 0:
                details = _run_nothink(
                    llm=llm,
                    tokenizer=tokenizer,
                    prompts=prompts,
                    samples=samples,
                    seed=inference_config.get("seed", 42),
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                    max_tokens=max_tokens,
                    chunk_size=inference_config.get("chunk_size", 512),
                )
                output_data = {
                    "metadata": {
                        "model": model,
                        "dataset": dataset_path,
                        "config_profile": config_profile,
                        "config": inference_config,
                        "eval_type": resolved_eval_type,
                        "samples": samples,
                        "max_tokens": max_tokens,
                        "run_name": run_name,
                        "nothink": True,
                    },
                    "analysis": None,
                    "evaluation": None,
                    "details": details,
                }
                save_json(data=output_data, file_path=str(file_path))
                log(f"    Saved to {file_path}")

            else:
                details = _run_twopass(
                    llm=llm,
                    tokenizer=tokenizer,
                    prompts=prompts,
                    max_think_tokens=max_think_tokens,  # type: ignore[arg-type]
                    overflow_suffix=overflow_suffix_str,
                    max_tokens=max_tokens,
                    samples=samples,
                    seed=inference_config.get("seed", 42),
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                    chunk_size=inference_config.get("chunk_size", 512),
                )

                # compute transition stats for metadata
                n_total = sum(len(task) for task in details)
                n_transition = sum(
                    1 for task in details for s in task if s["transition"]
                )

                output_data = {
                    "metadata": {
                        "model": model,
                        "dataset": dataset_path,
                        "config_profile": config_profile,
                        "config": inference_config,
                        "eval_type": resolved_eval_type,
                        "samples": samples,
                        "max_tokens": max_tokens,
                        "max_think_tokens": max_think_tokens,
                        "overflow_suffix_key": overflow_suffix,
                        "overflow_suffix": overflow_suffix_str,
                        "run_name": run_name,
                    },
                    "analysis": None,
                    "evaluation": None,
                    "details": details,
                }
                save_json(data=output_data, file_path=str(file_path))
                log(
                    f"    Saved to {file_path} "
                    f"(transition: {n_transition}/{n_total} = {n_transition / max(n_total, 1):.1%})",
                )

    else:
        log()
        log("Phase 1: Skipped (all datasets already generated)")

    # --- phase 2: evaluate all datasets ---
    log_header(f"PHASE 2: EVALUATION ({len(all_datasets)} datasets)")

    # summary file goes into the same root as the run (debug has its own subdir)
    summary_dir = Path(output) / "debug" if debug else Path(output)
    with json_results(
        directory=summary_dir,
        model=model,
        suffix=results_suffix,
    ) as model_results:
        eval_results: list[tuple[str, dict]] = []

        for i, (dataset_path, file_path, resolved_eval_type) in enumerate(
            all_datasets,
            start=1,
        ):
            log_step(i, len(all_datasets), f"{dataset_path}")

            try:
                data = load_json(file_path=str(file_path))

                eval_details: list[list[dict]] = data["details"]
                responses = [[s["text"] for s in task] for task in eval_details]

                dataset = load_dataset_records(
                    dataset=dataset_path,
                    default_dir="data",
                    split="test",
                    required_keys={"prompt"},
                )
                if debug:
                    dataset = dataset[:5]

                analysis = None
                breakdown = None
                if resolved_eval_type is not None:
                    log(f"    Evaluating {len(responses)} responses...")
                    with log_timer("evaluation"):
                        analysis, breakdown = evaluate(
                            responses=responses,
                            records=dataset,
                            eval_type=resolved_eval_type,
                            tokenizer=tokenizer,
                            details=eval_details,
                            dataset_name=Path(dataset_path).stem,
                        )

                    log(
                        f"    pass@1: {analysis['pass_at_1']:.2%}, "
                        f"pass@{analysis['k']}: {analysis['pass_at_k']:.2%} "
                        f"(n={analysis['total']})",
                    )
                    eval_results.append((dataset_path, analysis))

                    dataset_key = f"{dataset_path}_{config_profile}_{run_name}"
                    token_stats = aggregate_token_counts(
                        details=eval_details or data.get("token_counts", []),
                    )
                    model_results[dataset_key] = {
                        **analysis,
                        "token_statistics": token_stats,
                    }

                else:
                    log("    Skipped (no eval type specified)")

                data["analysis"] = analysis
                data["breakdown"] = breakdown
                data["metadata"]["eval_type"] = resolved_eval_type
                save_json(data=data, file_path=str(file_path))
                log("    Saved.")

            except Exception as e:
                log(f"    ERROR: {e}")
                traceback.print_exc()

    # --- summary ---
    log_header("SUMMARY")
    log(f"Model: {model}")
    log(f"Run: {run_name} ({mode_str})")
    log(f"Datasets evaluated: {len(eval_results)}/{len(all_datasets)}")
    log()
    if eval_results:
        log("Results:")
        for dataset_path, analysis in eval_results:
            stats = analysis["statistics"]
            log(
                f"  {dataset_path}: "
                f"pass@1={analysis['pass_at_1']:.2%}, "
                f"pass@{analysis['k']}={analysis['pass_at_k']:.2%}, "
                f"valid_reasoning={stats['valid_reasoning_rate']:.1%}, "
                f"missing={stats['missing_reasoning_rate']:.1%}",
            )
    log()
    log("Done.")
