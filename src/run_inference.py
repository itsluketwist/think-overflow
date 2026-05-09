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
    text, stop_reason, truncated, transition, prompt_tokens_1, completion_tokens_1,
    prompt_tokens_2, completion_tokens_2. Pass 2 keys are None for onepass.
    transition is always False for onepass (no forced cap).
    truncated is True when stop_reason is "length" (hit the token limit).
    """
    n_prompts = len(prompts)

    log("    Generating responses (onepass)...")
    with log_timer("generation"):
        responses, stop_reasons, prompt_toks, completion_toks = prompt_llm(
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
    for i in range(n_prompts):
        task_details: list[dict] = []
        for j in range(samples):
            stop_reason = stop_reasons[i][j]
            task_details.append(
                {
                    "text": responses[i][j],
                    "stop_reason": stop_reason,
                    "truncated": stop_reason == "length",  # hit model token limit
                    "transition": False,  # no forced cap in onepass
                    "prompt_tokens_1": prompt_toks[i],
                    "completion_tokens_1": completion_toks[i][j],
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
    answer_tokens: int | None,
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
    text, stop_reason, truncated, transition, prompt_tokens_1, completion_tokens_1,
    prompt_tokens_2, completion_tokens_2.
    transition=True means the reasoning was forcibly capped (pass 1 hit max_think_tokens).
    truncated=True means pass 2 answer generation hit the token limit.
    """
    n_prompts = len(prompts)
    # get model tag info once for stop condition and response reconstruction
    model_info = thinkpack.get_model_info(tokenizer)

    # --- pass 1: generate reasoning, stopping at close tag or max_think_tokens ---
    log("    Pass 1: generating capped reasoning...")
    with log_timer("pass 1"):
        (
            step1_responses,
            step1_reasons,
            p1_prompt_toks,
            p1_reason_toks,
        ) = prompt_llm(
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
        1
        for task_reasons in step1_reasons
        for reason in task_reasons
        if reason == "length"
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

    for step1_task_responses, step1_task_reasons in zip(step1_responses, step1_reasons):
        for response, reason in zip(step1_task_responses, step1_task_reasons):
            transition = bool(reason == "length")
            think_prefix = response.strip() + (overflow_suffix if transition else "")
            all_think_prefixes_flat.append(think_prefix)
            all_transition_flat.append(transition)

    # expand prompts to match flattened (prompt × sample) pairs
    flat_prompts = [p for p in prompts for _ in range(samples)]

    # --- pass 2: generate answers given the completed reasoning block ---
    log("    Pass 2: generating answers from reasoning...")
    with log_timer("pass 2"):
        (
            step2_responses,
            step2_reasons,
            p2_prompt_toks,
            p2_answer_toks,
        ) = prompt_llm(
            llm=llm,
            tokenizer=tokenizer,
            prompts=flat_prompts,
            samples=1,
            think_prefixes=all_think_prefixes_flat,
            seed=seed,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            max_tokens=answer_tokens,
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

            answer = step2_responses[k][0]
            stop_reason = step2_reasons[k][0]
            think_prefix = all_think_prefixes_flat[k]

            # reconstruct the full <think>...</think>answer for storage and evaluation
            full_text = f"{model_info.open_tag}\n{think_prefix}\n{model_info.close_tag}\n{answer}"

            task_details.append(
                {
                    "text": full_text,
                    "stop_reason": stop_reason,
                    "truncated": stop_reason == "length",  # answer generation hit limit
                    "transition": all_transition_flat[
                        k
                    ],  # reasoning was forcibly capped
                    "prompt_tokens_1": p1_prompt_toks[i],
                    "completion_tokens_1": p1_reason_toks[i][j],
                    "prompt_tokens_2": p2_prompt_toks[k],
                    "completion_tokens_2": p2_answer_toks[k][0],
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
    onepass: bool,
    max_think_tokens: int | None,
    overflow_suffix: str,
    run_name: str | None,
    update: bool,
    debug: bool = False,
) -> None:
    """Run two-pass or onepass inference with explicit configuration arguments.

    The overflow_suffix argument is a key from _OVERFLOW_SUFFIXES (e.g. 'formal'),
    not the resolved string. Mutually exclusive: either onepass=True or
    max_think_tokens must be set.
    """
    # resolve suffix key to the actual string appended to truncated reasoning
    overflow_suffix_str: str = _OVERFLOW_SUFFIXES[overflow_suffix]

    # default run name encodes the mode: "onepass" or "mt{cap}_{suffix}"
    if onepass:
        run_name = run_name or "onepass"
    else:
        run_name = run_name or f"mt{max_think_tokens}_{overflow_suffix}"

    log_header("INFERENCE")
    log(f"Model: {model}")
    log(f"Config: {config_file} [{config_profile}]")
    log(f"Mode: {'onepass (unconstrained)' if onepass else 'two-pass overflow'}")
    if not onepass:
        log(f"Max think tokens: {max_think_tokens}")
        log(f"Overflow suffix: {overflow_suffix} ({repr(overflow_suffix_str)})")
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

    # pass 2 answer budget: max_tokens from the config (null = no cap, generate until EOS)
    answer_tokens: int | None = inference_config.get("max_tokens")
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

    log(f"Answer tokens (pass 2 max): {answer_tokens}")
    log(f"Samples: {samples}")
    log(f"Temperature: {temperature if temperature is not None else 'default (1.0)'}")
    log(f"Top-p: {top_p if top_p is not None else 'default (1.0)'}")
    log(f"Top-k: {top_k if top_k is not None else 'default (-1)'}")

    if debug:
        # debug runs go to a fixed directory so they never pollute real results
        output_dir = Path(output) / "debug" / model
    else:
        # onepass and two-pass results are kept in separate directories
        output_subdir = "onepass" if onepass else "twopass"
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
            )

            output_data: dict[str, Any]
            if onepass:
                details = _run_onepass(
                    llm=llm,
                    tokenizer=tokenizer,
                    prompts=prompts,
                    samples=samples,
                    seed=inference_config.get("seed", 42),
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                    max_tokens=answer_tokens,
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
                        "run_name": run_name,
                        "onepass": True,
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
                    answer_tokens=answer_tokens,
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
                        "max_think_tokens": max_think_tokens,
                        "overflow_suffix_key": overflow_suffix,
                        "overflow_suffix": overflow_suffix_str,
                        "answer_tokens": answer_tokens,
                        "run_name": run_name,
                        "transition_count": n_transition,
                        "transition_rate": n_transition / max(n_total, 1),
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
    results_suffix = "_onepass" if onepass else "_twopass"
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

                # support both the new unified "details" format and legacy separate keys
                eval_details: list[list[dict]] | None
                if "details" in data:
                    eval_details = data["details"]
                    responses = [[s["text"] for s in task] for task in eval_details]
                else:
                    # legacy format: responses=[text, finish_reason]
                    raw = data["responses"]
                    responses = [[entry[0] for entry in task] for task in raw]
                    eval_details = None

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
    mode_str = (
        "onepass" if onepass else f"twopass (max_think_tokens={max_think_tokens})"
    )
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
