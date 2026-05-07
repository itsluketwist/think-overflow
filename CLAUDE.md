# think-overflow Project Instructions

This repository contains the code for the **think-overflow** method: two-pass capped-reasoning inference to handle the reasoning overflow problem in thinking-mode LLMs.

Thinking-mode models (Qwen3, OLMo-Think, Nemotron) may spend their entire token budget on reasoning without ever producing an answer. Think-overflow caps the reasoning phase at `max_think_tokens`, appends a recovery suffix to truncated traces, then feeds the completed thinking block back into the model to generate an answer.

## Known Issues

- vLLM progress bar shows incorrect total with `n>1` samples — harmless, can be ignored
- `transformers>=5.0.0` breaks vLLM 0.7.3 — pin `transformers<5.0.0` in requirements
- PyTorch 2.4+ prints a NCCL process group warning on exit — harmless

## Output Structure

- Single-pass results: `output/onepass/{model}/{dataset}_{config}_{run}.json`
- Two-pass results: `output/twopass/{model}/{dataset}_{config}_{run}.json`
- Per-model onepass summary: `output/{model}_onepass.json`
- Per-model two-pass summary: `output/{model}_twopass.json`
- Pre-computed token budget stats: `output/token_stats/{series_key}/{dataset}.json`
- Figures: `figures/`

## Code Style Notes

- Comments must be concise and fit on a single line

## Pipeline Overview

- One CLI entry point: `run`
- Two modes: `--onepass` (unconstrained, no token cap) and `--max-think-tokens N` (two-pass overflow)
- Three config profiles in `config/inference.yaml`: `greedymax` (greedy, 32k), `greedy` (greedy, 28k), `default` (model-recommended sampling, 28k, samples=3)
- Base models (with per-model default sampling params) registered in `config/models.yaml`
- HPC job submission via `scripts/submit_job.sh`; mode auto-derived from whether `max_think_tokens` is set

## Overflow Recovery Suffixes

Controlled via `--overflow-suffix` flag:

- `base`: nothing appended (pure truncation)
- `truncated`: appends " [reasoning truncated]"
- `formal`: appends "... I have to stop thinking and answer now."
- `human`: appends "... oops, I really need to stop thinking and to answer."

## Data Format

- Inference datasets: `data/{type}/*.jsonl` — required key: `prompt`; type auto-detected from directory (`code`, `math`, `reasoning`, `crux`)
- Download all datasets by running `data/download.ipynb`

## External Dependencies

- `llm_cgr` (llm-codegen-research): utility package — `save_json(data, path)`, `save_jsonl(data, path)`, `load_jsonl(file_path)`
