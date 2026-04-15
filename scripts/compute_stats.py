"""Compute and persist token budget statistics needed to reproduce violin plots.

Run from the repository root:
    python scripts/compute_stats.py           # skip already-computed entries
    python scripts/compute_stats.py --update  # recompute everything

Results are saved to output/stats/{series_key}/{dataset_label}.json.
Add new models to SERIES (and DATASETS if needed) then re-run to extend the dataset.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from transformers import AutoTokenizer


# add src to the path so we can import the response parser
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from evaluate.parser import parse_response


# ---------------------------------------------------------------------------
# Configuration — edit these to add or remove models and datasets
# ---------------------------------------------------------------------------

# series: each entry is a (model, decoding strategy) combination.
# file_suffix is appended to each dataset's base file stem when looking for
# the inference output (e.g. suffix "_steer" turns "evalplus_greedy" →
# "evalplus_greedy_steer").
SERIES: dict[str, dict] = {
    "qwen3_greedy": {
        "model_key": "qwen3-8b",
        "hf_path": "Qwen/Qwen3-8B",
        "olmo_style": False,
        "file_suffix": "",
    },
    "qwen3_prompt_strict": {
        "model_key": "qwen3-8b",
        "hf_path": "Qwen/Qwen3-8B",
        "olmo_style": False,
        "file_suffix": "_prompt-strict",
    },
    "qwen3_steer_strict": {
        "model_key": "qwen3-8b",
        "hf_path": "Qwen/Qwen3-8B",
        "olmo_style": False,
        "file_suffix": "_steer-strict",
    },
    "olmo_greedy": {
        "model_key": "olmo-3-7b-think",
        "hf_path": "allenai/OLMo-3-7B-Think",
        "olmo_style": True,
        "file_suffix": "",
    },
    "olmo_prompt_strict": {
        "model_key": "olmo-3-7b-think",
        "hf_path": "allenai/OLMo-3-7B-Think",
        "olmo_style": True,
        "file_suffix": "_prompt-strict",
    },
    "olmo_steer_strict": {
        "model_key": "olmo-3-7b-think",
        "hf_path": "allenai/OLMo-3-7B-Think",
        "olmo_style": True,
        "file_suffix": "_steer-strict",
    },
    "nemotron_greedy": {
        "model_key": "nemotron-7b",
        "hf_path": "nvidia/OpenReasoning-Nemotron-7B",
        "olmo_style": False,
        "file_suffix": "",
    },
    "nemotron_prompt_strict": {
        "model_key": "nemotron-7b",
        "hf_path": "nvidia/OpenReasoning-Nemotron-7B",
        "olmo_style": False,
        "file_suffix": "_prompt-strict",
    },
    "nemotron_steer_strict": {
        "model_key": "nemotron-7b",
        "hf_path": "nvidia/OpenReasoning-Nemotron-7B",
        "olmo_style": False,
        "file_suffix": "_steer-strict",
    },
    "code_nemotron_greedy": {
        "model_key": "code-nemotron-7b",
        "hf_path": "nvidia/OpenCodeReasoning-Nemotron-7B",
        "olmo_style": False,
        "file_suffix": "",
    },
    "code_nemotron_prompt_strict": {
        "model_key": "code-nemotron-7b",
        "hf_path": "nvidia/OpenCodeReasoning-Nemotron-7B",
        "olmo_style": False,
        "file_suffix": "_prompt-strict",
    },
    "code_nemotron_steer_strict": {
        "model_key": "code-nemotron-7b",
        "hf_path": "nvidia/OpenCodeReasoning-Nemotron-7B",
        "olmo_style": False,
        "file_suffix": "_steer-strict",
    },
}

# datasets: label → (inference file stem, prompt jsonl path relative to data/)
DATASETS: dict[str, tuple[str, str]] = {
    "evalplus": ("evalplus_greedy", "code/evalplus.jsonl"),
    "livecodebench": ("livecodebench_greedy", "code/livecodebench.jsonl"),
    "cruxeval_i": ("cruxeval_i_greedy", "crux/cruxeval_i.jsonl"),
    "cruxeval_o": ("cruxeval_o_greedy", "crux/cruxeval_o.jsonl"),
    "bigcodebench": ("bigcodebench_greedy", "code/bigcodebench.jsonl"),
    "gsm8k": ("gsm8k_greedy", "math/gsm8k.jsonl"),
    "gpqa": ("gpqa_greedy", "reasoning/gpqa.jsonl"),
}

# generation cap used for all runs — needed to compute budget percentages
MAX_TOKENS: int = 32768

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).parent.parent
INFER_DIR = REPO_ROOT / "output" / "infer"
DATA_DIR = REPO_ROOT / "data"
OUTPUT_DIR = REPO_ROOT / "output" / "stats"


# ---------------------------------------------------------------------------
# Core computation
# ---------------------------------------------------------------------------


def compute_series_dataset(
    series_key: str,
    dataset_label: str,
    tokenizer: AutoTokenizer,
    update: bool,
) -> None:
    """Compute token budget statistics for one (series, dataset) pair and save to disk.

    Skips if the output file already exists and update is False.
    Truncation is inferred structurally (open tag without close tag) rather than
    from finish_reason, matching the notebook's approach.
    """
    cfg = SERIES[series_key]
    ds_file, ds_jsonl = DATASETS[dataset_label]
    file_stem = f"{ds_file}{cfg['file_suffix']}"
    out_path = OUTPUT_DIR / series_key / f"{dataset_label}.json"

    # skip if output already exists and we are not forcing an update
    if out_path.exists() and not update:
        print(f"  [skip] {series_key}/{dataset_label} (already exists)")
        return

    source_path = INFER_DIR / cfg["model_key"] / f"{file_stem}.json"
    if not source_path.exists():
        print(f"  [skip] {series_key}/{dataset_label} — missing {source_path}")
        return

    # load inference results and original prompts
    with open(source_path) as f:
        data = json.load(f)

    jsonl_path = DATA_DIR / ds_jsonl
    with open(jsonl_path) as f:
        prompts = [json.loads(line)["prompt"] for line in f]

    token_counts: list[int] = []
    truncated: list[bool] = []
    overflow: list[bool] = []
    budget_pct: list[float] = []

    for task_idx, sample_responses in enumerate(data["responses"]):
        # each sample is [response_text, finish_reason]; greedy = one sample per task
        response, finish_reason = sample_responses[0][0], sample_responses[0][1]
        parsed = parse_response(
            response=response,
            olmo_style=cfg["olmo_style"],
        )

        # count tokens in the reasoning block; 0 if no reasoning present
        n_tokens = len(tokenizer.encode(parsed.reasoning))
        token_counts.append(n_tokens)

        # truncated = reasoning block opened but closing tag never appeared
        is_truncated = parsed.has_reasoning_block and not parsed.has_valid_reasoning
        truncated.append(is_truncated)

        # overflow = generation hit the token limit (finish_reason from the model)
        overflow.append(finish_reason == "length")

        # available budget = generation cap minus tokens consumed by the prompt
        prompt_tokens = len(tokenizer.encode(prompts[task_idx]))
        available = MAX_TOKENS - prompt_tokens

        if is_truncated:
            # pin at 100% so truncated responses hit the budget ceiling visually
            pct = 100.0
        else:
            pct = min(100.0 * n_tokens / available, 100.0)
        budget_pct.append(pct)

    # pass@1 is pre-computed and stored in the inference result's analysis block
    pass_at_1: float = data.get("analysis", {}).get("pass_at_1", float("nan"))

    # summary line matching the notebook's output format for easy cross-checking
    n_trunc = sum(truncated)
    n_overflow = sum(overflow)
    print(
        f"  {series_key} / {dataset_label}: "
        f"n={len(token_counts)}, "
        f"median={int(np.median(token_counts)):,} tokens, "
        f"median budget={np.median(budget_pct):.1f}%, "
        f"pass@1={100 * pass_at_1:.1f}%, "
        f"truncated={n_trunc} ({100 * n_trunc / len(truncated):.1f}%), "
        f"overflow={n_overflow} ({100 * n_overflow / len(overflow):.1f}%)"
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(
            {
                "series_key": series_key,
                "model_key": cfg["model_key"],
                "dataset": dataset_label,
                "token_counts": token_counts,
                "truncated": truncated,
                "overflow": overflow,
                "budget_pct": budget_pct,
                "pass_at_1": pass_at_1,
                "source_file": str(source_path.relative_to(REPO_ROOT)),
                "source_mtime": source_path.stat().st_mtime,
            },
            f,
        )


def main() -> None:
    """Load tokenizers then compute statistics for every (series, dataset) pair."""
    parser = argparse.ArgumentParser(
        description="Compute token budget statistics for violin plots.",
    )
    parser.add_argument(
        "--update",
        action="store_true",
        help="recompute all entries, even if output files already exist",
    )
    args = parser.parse_args()

    # load one tokenizer per unique hf_path — many series share the same model
    print("loading tokenizers...")
    tokenizers: dict[str, AutoTokenizer] = {}
    for series_key, cfg in SERIES.items():
        hf_path = cfg["hf_path"]
        if hf_path not in tokenizers:
            tokenizers[hf_path] = AutoTokenizer.from_pretrained(hf_path)
            print(f"  loaded {hf_path}")

    print(f"\ncomputing stats (update={args.update})...")
    for series_key, cfg in SERIES.items():
        tokenizer = tokenizers[cfg["hf_path"]]
        for dataset_label in DATASETS:
            compute_series_dataset(
                series_key=series_key,
                dataset_label=dataset_label,
                tokenizer=tokenizer,
                update=args.update,
            )

    print("\ndone.")


if __name__ == "__main__":
    main()
