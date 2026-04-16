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

from llm_cgr import load_json, save_json
from vllm import LLM

from src.evaluate import evaluate
from src.evaluate.statistics import (
    aggregate_baseline_token_counts,
    aggregate_token_counts,
)
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


def _build_pass2_prefix(
    response: str,
    olmo_style: bool,
    overflow_suffix: str,
    truncated: bool,
) -> str:
    """Construct the full <think>...</think> prefix for Pass 2.

    Reconstructs a canonical thinking block from the Pass 1 response.
    OLMo models emit raw content (template injects the opening tag), so we
    add it back explicitly. For non-OLMo models the response should already
    start with <think>, but we wrap it if not to handle edge cases.
    The overflow_suffix is appended only when the reasoning was truncated.

    Returns a string of the form '<think>\\n{reasoning}{suffix}\\n</think>\\n'.
    """
    if olmo_style or not response.startswith("<think>"):
        # olmo: template injects <think>, response is raw content
        # non-olmo edge case: model didn't emit the tag, wrap manually
        block = "<think>\n" + response
    else:
        # non-olmo: response already starts with <think>
        block = response

    # append the overflow suffix only if reasoning was cut short by the token cap
    suffix = overflow_suffix if truncated else ""
    return block + suffix + "\n</think>\n"


def _run_single_pass(
    llm: LLM,
    prompts: list[str],
    samples: int,
    seed: int,
    temperature: float,
    top_p: float,
    top_k: int,
    max_tokens: int,
    chunk_size: int,
) -> tuple[
    list[list[list[Any]]],  # packed_responses[task][sample] = [text, finish_reason]
    list[list[dict]],  # token_counts[task][sample] = {prompt_tokens, completion_tokens}
]:
    """Run single-pass standard inference over a list of prompts.

    Collects prompt and completion token counts for each sample in the same
    [task][sample] shape as _run_two_pass so the rest of the pipeline is unchanged.

    Returns (packed_responses, token_counts).
    """
    n_prompts = len(prompts)

    log("    Generating responses (single pass)...")
    with log_timer("generation"):
        responses, finish_reasons, prompt_toks, completion_toks = prompt_llm(
            llm=llm,
            prompts=prompts,
            samples=samples,
            prefixes=None,
            seed=seed,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            max_tokens=max_tokens,
            min_tokens=1,
            stop=None,
            chunk_size=chunk_size,
        )

    packed_responses: list[list[list[Any]]] = []
    token_counts: list[list[dict]] = []

    for i in range(n_prompts):
        task_responses: list[list[Any]] = []
        task_tokens: list[dict] = []

        for j in range(samples):
            task_responses.append([responses[i][j], finish_reasons[i][j]])
            task_tokens.append(
                {
                    "prompt_tokens": prompt_toks[i],
                    "completion_tokens": completion_toks[i][j],
                },
            )

        packed_responses.append(task_responses)
        token_counts.append(task_tokens)

    return packed_responses, token_counts


def _run_two_pass(
    llm: LLM,
    prompts: list[str],
    olmo_style: bool,
    max_think_tokens: int,
    overflow_suffix: str,
    answer_tokens: int,
    samples: int,
    seed: int,
    temperature: float,
    top_p: float,
    top_k: int,
    chunk_size: int,
) -> tuple[
    list[list[list[Any]]],  # packed_responses[task][sample] = [text, finish_reason]
    list[list[bool]],  # truncated[task][sample]
    list[list[dict]],  # token_counts[task][sample] = {pass1_prompt, ...}
]:
    """Run two-pass capped reasoning inference over a list of prompts.

    Pass 1 generates reasoning up to max_think_tokens (stopping at </think>
    if the model closes its thinking naturally before the cap). Truncated
    samples get overflow_suffix appended before closing </think>. Pass 2
    generates the answer given the full reasoning context.

    Returns (packed_responses, truncated_flags, token_counts).
    """
    n_prompts = len(prompts)

    # --- pass 1: generate reasoning, stopping at </think> or max_think_tokens ---
    log("    Pass 1: generating capped reasoning...")
    with log_timer("pass 1"):
        (
            step1_responses,
            step1_reasons,
            p1_prompt_toks,
            p1_reason_toks,
        ) = prompt_llm(
            llm=llm,
            prompts=prompts,
            samples=samples,
            prefixes=None,
            seed=seed,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            max_tokens=max_think_tokens,
            min_tokens=1,
            stop=["</think>"],
            chunk_size=chunk_size,
        )

    # count how many reasoning traces were cut short by the token cap
    truncation_count = sum(
        1
        for task_reasons in step1_reasons
        for reason in task_reasons
        if reason == "length"
    )
    log(
        f"    Truncated: {truncation_count}/{n_prompts * samples} "
        f"({truncation_count / max(n_prompts * samples, 1):.1%})",
    )

    # --- build pass 2 prefixes: one per (prompt, sample) pair ---
    # flattened in row-major order: all samples for prompt 0, then prompt 1, ...
    all_prefixes_flat: list[str] = []
    all_truncated_flat: list[bool] = []

    for step1_task_responses, step1_task_reasons in zip(step1_responses, step1_reasons):
        for response, reason in zip(step1_task_responses, step1_task_reasons):
            truncated = reason == "length"
            prefix = _build_pass2_prefix(
                response=response,
                olmo_style=olmo_style,
                overflow_suffix=overflow_suffix,
                truncated=truncated,
            )
            all_prefixes_flat.append(prefix)
            all_truncated_flat.append(truncated)

    # expand prompts to match flattened (prompt × sample) pairs
    flat_prompts = [p for p in prompts for _ in range(samples)]

    # --- pass 2: generate answers given the complete reasoning context ---
    # prompt_llm prepends each prefix to the stored response, so the output is
    # "<think>\n{reasoning}\n</think>\n{answer}" — compatible with evaluate()
    log("    Pass 2: generating answers from reasoning...")
    with log_timer("pass 2"):
        (
            step2_responses,
            step2_reasons,
            p2_prompt_toks,
            p2_answer_toks,
        ) = prompt_llm(
            llm=llm,
            prompts=flat_prompts,
            samples=1,
            prefixes=all_prefixes_flat,
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
    packed_responses: list[list[list[Any]]] = []
    truncated_flags: list[list[bool]] = []
    token_counts: list[list[dict]] = []

    for i in range(n_prompts):
        task_responses: list[list[Any]] = []
        task_truncated: list[bool] = []
        task_tokens: list[dict] = []

        for j in range(samples):
            # flat index for this (prompt i, sample j) pair
            k = i * samples + j

            # step2_responses[k] is a 1-element list (samples=1 in pass 2)
            full_response = step2_responses[k][0]
            finish_reason = step2_reasons[k][0]

            task_responses.append([full_response, finish_reason])
            task_truncated.append(all_truncated_flat[k])
            task_tokens.append(
                {
                    # tokens in the original prompt (without reasoning)
                    "pass1_prompt": p1_prompt_toks[i],
                    # reasoning tokens generated in pass 1 (output cost)
                    "pass1_reasoning": p1_reason_toks[i][j],
                    # prompt + reasoning prefix input to pass 2
                    "pass2_prompt": p2_prompt_toks[k],
                    # answer tokens generated in pass 2 (output cost)
                    "pass2_answer": p2_answer_toks[k][0],
                    # total tokens generated across both passes (effective cost)
                    "total_generated": p1_reason_toks[i][j] + p2_answer_toks[k][0],
                },
            )

        packed_responses.append(task_responses)
        truncated_flags.append(task_truncated)
        token_counts.append(task_tokens)

    return packed_responses, truncated_flags, token_counts


def run_inference(
    model: str,
    model_config: str,
    config_file: str,
    config_profile: str,
    output: str,
    datasets: str,
    eval_type: str | None,
    baseline: bool,
    max_think_tokens: int | None,
    overflow_suffix: str,
    run_name: str | None,
    update: bool,
) -> None:
    """Run two-pass or single-pass inference with explicit configuration arguments.

    The overflow_suffix argument is a key from _OVERFLOW_SUFFIXES (e.g. 'formal'),
    not the resolved string. Mutually exclusive: either baseline=True or
    max_think_tokens must be set.
    """
    # resolve suffix key to the actual string appended to truncated reasoning
    overflow_suffix_str: str = _OVERFLOW_SUFFIXES[overflow_suffix]

    # default run name encodes the mode: "baseline" or "mt{cap}_{suffix}"
    if baseline:
        run_name = run_name or "baseline"
    else:
        run_name = run_name or f"mt{max_think_tokens}_{overflow_suffix}"

    log_header("INFERENCE")
    log(f"Model: {model}")
    log(f"Config: {config_file} [{config_profile}]")
    log(f"Mode: {'baseline (single-pass)' if baseline else 'two-pass overflow'}")
    if not baseline:
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

    # olmo_style flag: olmo models inject the opening <think> tag in their chat template,
    # so reasoning responses are raw content rather than wrapped in <think>...</think>
    olmo_style: bool = model_cfg.get("olmo_style", False)
    # pass 2 uses the full max_tokens from the config (typically 32k)
    answer_tokens: int = inference_config.get("max_tokens", 32768)
    samples: int = inference_config.get("samples", 1)

    log(f"OLMo style: {olmo_style}")
    log(f"Answer tokens (pass 2 max): {answer_tokens}")
    log(f"Samples: {samples}")
    log(f"Temperature: {inference_config.get('temperature', 0.0)}")
    log(f"Top-p: {inference_config.get('top_p', 1.0)}")
    log(f"Top-k: {inference_config.get('top_k', -1)}")

    # baseline results and two-pass results are kept in separate directories
    # so they can be compared without ambiguity
    output_subdir = "full_baseline" if baseline else "full_transition"
    output_dir = Path(output) / output_subdir / model
    output_dir.mkdir(parents=True, exist_ok=True)
    log(f"Output directory: {output_dir}")

    # parse all dataset entries and determine which need generation
    dataset_entries = [parse_dataset_entry(d) for d in datasets.split(",") if d]
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
            log(f"    Loaded {len(dataset)} prompts")

            prompts = construct_inference_prompts(
                records=dataset,
                eval_type=resolved_eval_type,
            )

            output_data: dict[str, Any]
            if baseline:
                packed_responses, token_counts = _run_single_pass(
                    llm=llm,
                    prompts=prompts,
                    samples=samples,
                    seed=inference_config.get("seed", 42),
                    temperature=inference_config.get("temperature", 0.0),
                    top_p=inference_config.get("top_p", 1.0),
                    top_k=inference_config.get("top_k", -1),
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
                        "olmo_style": olmo_style,
                        "baseline": True,
                    },
                    "analysis": None,
                    "evaluation": None,
                    "responses": packed_responses,
                    "token_counts": token_counts,
                }
                save_json(data=output_data, file_path=str(file_path))
                log(f"    Saved to {file_path}")

            else:
                packed_responses, truncated_flags, token_counts = _run_two_pass(
                    llm=llm,
                    prompts=prompts,
                    olmo_style=olmo_style,
                    max_think_tokens=max_think_tokens,  # type: ignore[arg-type]
                    overflow_suffix=overflow_suffix_str,
                    answer_tokens=answer_tokens,
                    samples=samples,
                    seed=inference_config.get("seed", 42),
                    temperature=inference_config.get("temperature", 0.0),
                    top_p=inference_config.get("top_p", 1.0),
                    top_k=inference_config.get("top_k", -1),
                    chunk_size=inference_config.get("chunk_size", 512),
                )

                # compute truncation stats for metadata
                n_total = sum(len(task) for task in truncated_flags)
                n_truncated = sum(1 for task in truncated_flags for t in task if t)

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
                        "olmo_style": olmo_style,
                        "truncation_count": n_truncated,
                        "truncation_rate": n_truncated / max(n_total, 1),
                    },
                    "analysis": None,
                    "evaluation": None,
                    "responses": packed_responses,
                    "truncated": truncated_flags,
                    "token_counts": token_counts,
                }
                save_json(data=output_data, file_path=str(file_path))
                log(
                    f"    Saved to {file_path} "
                    f"(truncation: {n_truncated}/{n_total} = {n_truncated / max(n_total, 1):.1%})",
                )

    else:
        log()
        log("Phase 1: Skipped (all datasets already generated)")

    # --- phase 2: evaluate all datasets ---
    log_header(f"PHASE 2: EVALUATION ({len(all_datasets)} datasets)")

    with json_results(
        directory=output,
        model=model,
    ) as model_results:
        eval_results: list[tuple[str, dict]] = []

        for i, (dataset_path, file_path, resolved_eval_type) in enumerate(
            all_datasets,
            start=1,
        ):
            log_step(i, len(all_datasets), f"{dataset_path}")

            try:
                data = load_json(file_path=str(file_path))
                raw_responses = data["responses"]

                # unpack [text, finish_reason] pairs stored per sample
                responses = [[entry[0] for entry in task] for task in raw_responses]
                finish_reasons = [
                    [entry[1] for entry in task] for task in raw_responses
                ]

                dataset = load_dataset_records(
                    dataset=dataset_path,
                    default_dir="data",
                    split="test",
                    required_keys={"prompt"},
                )

                is_two_pass = not data["metadata"].get("baseline", False)
                analysis = None
                breakdown = None
                if resolved_eval_type is not None:
                    log(f"    Evaluating {len(responses)} responses...")
                    # two-pass: stored truncated_flags are pass 1 cap-hit booleans
                    # baseline: derive equivalent from finish_reasons (True = any overflow)
                    eval_truncated_flags: list[list[bool]] = data.get("truncated") or [
                        [r == "length" for r in task] for task in finish_reasons
                    ]
                    with log_timer("evaluation"):
                        analysis, breakdown = evaluate(
                            responses=responses,
                            records=dataset,
                            eval_type=resolved_eval_type,
                            olmo_style=olmo_style,
                            finish_reasons=finish_reasons,
                            truncated_flags=eval_truncated_flags,
                            two_pass=is_two_pass,
                            dataset_name=Path(dataset_path).stem,
                        )

                    log(
                        f"    pass@1: {analysis['pass_at_1']:.2%}, "
                        f"pass@{analysis['k']}: {analysis['pass_at_k']:.2%} "
                        f"(n={analysis['total']})",
                    )
                    eval_results.append((dataset_path, analysis))

                    dataset_key = f"{dataset_path}_{config_profile}_{run_name}"
                    # use the appropriate aggregator depending on the run mode
                    if not is_two_pass:
                        token_stats = aggregate_baseline_token_counts(
                            token_counts=data["token_counts"],
                        )
                    else:
                        token_stats = aggregate_token_counts(
                            token_counts=data["token_counts"],
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
    log(f"Run: {run_name} (max_think_tokens={max_think_tokens})")
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
                f"reasoning={stats['any_reasoning']:.1%}, "
                f"valid={stats['valid_reasoning']:.1%}",
            )
    log()
    log("Done.")
