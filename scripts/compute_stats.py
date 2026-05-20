"""Compute and persist token budget statistics needed to reproduce violin plots.

Run from the repository root:
    python scripts/compute_stats.py           # skip already-computed entries
    python scripts/compute_stats.py --update  # recompute everything

Results are saved to output/token_stats/{model_key}.json — one file per model,
with all datasets nested under a "datasets" key. Each dataset key encodes both
the dataset name and the token budget (e.g. "evalplus_greedy_mx32768"), so all
budget sizes (8k, 16k, 32k) coexist in the same file.
Add new models or datasets then re-run to extend.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import thinkpack
from llm_cgr import save_json
from transformers import AutoTokenizer


sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


# ---------------------------------------------------------------------------
# Configuration — edit these to add or remove models and datasets
# ---------------------------------------------------------------------------

# series: each entry is a (model, run type) combination.
# file_suffix is appended to each dataset's base file stem to form the
# inference filename (e.g. "_onepass" turns "evalplus_greedy_mx32768" →
# "evalplus_greedy_mx32768_onepass").
SERIES: dict[str, dict] = {
    "qwen3_8b": {
        "model_key": "qwen3-8b",
        "hf_path": "Qwen/Qwen3-8B",
        "file_suffix": "_onepass",
    },
    "olmo_3_7b": {
        "model_key": "olmo-3-7b",
        "hf_path": "allenai/OLMo-3-7B-Think",
        "file_suffix": "_onepass",
    },
    "llama_r1_8b": {
        "model_key": "llama-r1-8b",
        "hf_path": "deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
        "file_suffix": "_onepass",
    },
    "ministral_8b": {
        "model_key": "ministral-8b",
        "hf_path": "mistralai/Ministral-3-8B-Reasoning-2512",
        "file_suffix": "_onepass",
    },
    "or_7b": {
        "model_key": "or-7b",
        "hf_path": "nvidia/OpenReasoning-Nemotron-7B",
        "file_suffix": "_onepass",
    },
    "ocr_7b": {
        "model_key": "ocr-7b",
        "hf_path": "nvidia/OpenCodeReasoning-Nemotron-1.1-7B",
        "file_suffix": "_onepass",
    },
}

# token budgets to compute stats for — each produces a separate run_key in the output
BUDGETS: list[int] = [32768, 16384, 8192]

# datasets: label → (inference file stem base, prompt jsonl path relative to data/)
# the budget is appended at runtime (e.g. "evalplus_greedy_mx" + "32768")
DATASETS: dict[str, tuple[str, str]] = {
    "evalplus": ("evalplus_greedy_mx", "code/evalplus.jsonl"),
    "livecodebench": ("livecodebench_greedy_mx", "code/livecodebench.jsonl"),
    "bigcodebench": ("bigcodebench_greedy_mx", "code/bigcodebench.jsonl"),
    "code_contests": ("code_contests_greedy_mx", "code/code_contests.jsonl"),
    "cruxeval_i": ("cruxeval_i_greedy_mx", "crux/cruxeval_i.jsonl"),
    "cruxeval_o": ("cruxeval_o_greedy_mx", "crux/cruxeval_o.jsonl"),
    "gsm8k": ("gsm8k_greedy_mx", "math/gsm8k.jsonl"),
    "math500": ("math500_greedy_mx", "math/math500.jsonl"),
    "gpqa": ("gpqa_greedy_mx", "reasoning/gpqa.jsonl"),
}


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).parent.parent
# inference results: output/onepass/{model}/{dataset}_{config}_{run}.json
INFER_DIR = REPO_ROOT / "output" / "onepass"
# per-model baseline summary files (pass_at_1 lives here): output/{model}_onepass.json
SUMMARY_DIR = REPO_ROOT / "output"
# stats output: output/token_stats/{model_key}.json — one file per model
OUTPUT_DIR = REPO_ROOT / "output" / "token_stats"


# ---------------------------------------------------------------------------
# Core computation
# ---------------------------------------------------------------------------


def load_pass_at_1(
    model_key: str,
    data: dict,
) -> float:
    """Look up pass@1 for this run from the per-model summary file.

    The summary file (output/{model}_onepass.json) stores results keyed by
    "{dataset}_{config}_{run_name}". Returns nan if the summary
    file or the key is missing.
    """
    # baseline summary file uses the _baseline suffix (written by run_inference with baseline=True)
    summary_path = SUMMARY_DIR / f"{model_key}_onepass.json"
    if not summary_path.exists():
        return float("nan")

    with open(summary_path) as f:
        summary = json.load(f)

    # reconstruct the summary key from the inference file's own metadata
    meta = data["metadata"]
    summary_key = f"{meta['dataset']}_{meta['config_profile']}_{meta['run_name']}"
    return summary.get(summary_key, {}).get("pass_at_1", float("nan"))


def compute_series_dataset(
    series_key: str,
    dataset_label: str,
    budget: int,
    tokenizer: AutoTokenizer,
) -> dict | None:
    """Compute token budget statistics for one (series, dataset, budget) triple.

    Returns None if the inference file is missing, otherwise returns a stats dict
    with keys: reasoning_token_counts, budget_pct, pass_at_1, max_tokens,
    source_file, source_mtime.
    """
    cfg = SERIES[series_key]
    ds_base, _ = DATASETS[dataset_label]

    # e.g. "evalplus_greedy_mx32768_onepass"
    file_stem = f"{ds_base}{budget}{cfg['file_suffix']}"
    source_path = INFER_DIR / cfg["model_key"] / f"{file_stem}.json"

    if not source_path.exists():
        print(f"  [skip] {series_key}/{dataset_label} — missing {source_path}")
        return None

    with open(source_path) as f:
        data = json.load(f)

    pass_at_1 = load_pass_at_1(model_key=cfg["model_key"], data=data)

    # max_tokens is the generation cap for this run — read from the file so the
    # script works correctly across runs with different budgets (e.g. 8k, 32k)
    max_tokens: int = data["metadata"]["max_tokens"]

    reasoning_token_counts: list[int] = []
    budget_pct: list[float] = []

    for task_details in data["details"]:
        # greedy runs have one sample per task; take the first
        sample = task_details[0]
        response = sample["text"]
        # add_generation_reasoning=True handles models where <think> is in the prompt
        # template (e.g. Qwen3, OLMo-Think), so thinkpack extracts reasoning correctly
        parsed = thinkpack.parse(
            response=response,
            tokenizer=tokenizer,
            add_generation_reasoning=True,
        )

        # count tokens without special tokens so the count matches the model's internal count
        n_tokens = len(
            tokenizer.encode(parsed.reasoning, add_special_tokens=False),
        )
        reasoning_token_counts.append(n_tokens)

        # no answer and generation hit the length cap → model used its full budget on reasoning
        no_answer = not parsed.answer and sample["stop_reason"] == "length"

        if no_answer:
            # pin at 100% so these responses hit the budget ceiling visually
            pct = 100.0
        else:
            # max_tokens controls new generated tokens only (prompt tokens are not counted),
            # so it is the correct denominator — no prompt subtraction needed
            pct = min(100.0 * n_tokens / max_tokens, 100.0)
        budget_pct.append(pct)

    n = len(reasoning_token_counts)
    print(
        f"  {series_key} / {dataset_label}: "
        f"n={n}, "
        f"median={int(np.median(reasoning_token_counts)):,} reasoning tokens, "
        f"median budget={np.median(budget_pct):.1f}%, "
        f"pass@1={100 * pass_at_1:.1f}%"
    )

    return {
        "reasoning_token_counts": reasoning_token_counts,
        "budget_pct": budget_pct,
        "pass_at_1": pass_at_1,
        "max_tokens": max_tokens,
        "source_file": str(source_path.relative_to(REPO_ROOT)),
        "source_mtime": source_path.stat().st_mtime,
    }


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
            tokenizers[hf_path] = AutoTokenizer.from_pretrained(
                pretrained_model_name_or_path=hf_path,
                fix_mistral_regex=True,
            )
            print(f"  loaded {hf_path}")

    print(f"\ncomputing stats (update={args.update})...")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for series_key, cfg in SERIES.items():
        tokenizer = tokenizers[cfg["hf_path"]]
        model_key = cfg["model_key"]
        out_path = OUTPUT_DIR / f"{model_key}.json"

        # load existing model file so we can skip already-computed datasets
        existing: dict[str, dict] = {}
        if out_path.exists() and not args.update:
            with open(out_path) as f:
                existing = json.load(f).get("datasets", {})

        datasets: dict[str, dict] = {}
        for budget in BUDGETS:
            for dataset_label in DATASETS:
                # use the full inference file stem as the key (e.g. "evalplus_greedy_mx32768")
                # so the key captures the dataset, config, and token budget
                ds_base, _ = DATASETS[dataset_label]
                run_key = f"{ds_base}{budget}"

                if run_key in existing and not args.update:
                    print(f"  [skip] {series_key}/{run_key} (already exists)")
                    datasets[run_key] = existing[run_key]
                    continue

                result = compute_series_dataset(
                    series_key=series_key,
                    dataset_label=dataset_label,
                    budget=budget,
                    tokenizer=tokenizer,
                )
                if result is not None:
                    datasets[run_key] = result

        # save all datasets for this model in one pretty-printed file
        save_json(
            data={
                "series_key": series_key,
                "model_key": model_key,
                "datasets": datasets,
            },
            file_path=str(out_path),
        )
        print(f"  saved {out_path.relative_to(REPO_ROOT)}")

    print("\ndone.")


if __name__ == "__main__":
    main()
