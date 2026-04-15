# think-overflow Project Instructions

This repository contains the code for the **think-overflow** method: two-pass capped-reasoning inference to handle the reasoning overflow problem in thinking-mode LLMs.

Thinking-mode models (Qwen3, OLMo-Think, Nemotron) may spend their entire token budget on reasoning without ever producing an answer. Think-overflow caps the reasoning phase at `max_think_tokens`, appends a recovery suffix to truncated traces, then feeds the completed thinking block back into the model to generate an answer.

## Known Issues

- vLLM progress bar shows incorrect total with `n>1` samples — harmless, can be ignored
- `transformers>=5.0.0` breaks vLLM 0.7.3 — pin `transformers<5.0.0` in requirements
- OLMo3 models have `olmo_style: true` in config — their chat templates inject an opening `<think>` tag automatically; a missing `</think>` in generated output causes response truncation
- PyTorch 2.4+ prints a NCCL process group warning on exit — harmless

## Output Structure

- Inference results: `output/infer/{model}/{dataset}_{config}_{run}.json`
- Per-model summary (all runs): `output/{model}.json`
- Pre-computed token budget stats: `output/stats/{series_key}/{dataset}.json`
- Figures: `figures/01/`, `figures/02/`

## Code Style Notes

- Comments must be concise and fit on a single line

## Pipeline Overview

- One CLI entry point: `infer`
- Two modes: `--baseline` (single-pass greedy) and `--max-think-tokens N` (two-pass overflow)
- Base models registered in `config/models.yaml`
- HPC job submission via `scripts/submit_job.sh`
- Config profiles in `config/inference.yaml`; use `greedy` for all experiments

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
