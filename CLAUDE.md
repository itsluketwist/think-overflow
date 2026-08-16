# think-overflow Project Instructions

This repository contains the code for the **think-overflow** method: two-pass capped-reasoning inference to handle the reasoning overflow problem in thinking-mode LLMs.

Thinking-mode models (Qwen3, OLMo-Think, Nemotron) may spend their entire token budget on reasoning without ever producing an answer. Think-overflow caps the reasoning phase at `max_think_tokens`, appends a recovery suffix to truncated traces, then feeds the completed thinking block back into the model to generate an answer.

## Known Issues

- vLLM progress bar shows incorrect total with `n>1` samples — harmless, can be ignored
- `transformers>=5.0.0` breaks vLLM — keep `transformers` pinned below it in `pyproject.toml`
- PyTorch 2.4+ prints a NCCL process group warning on exit — harmless

## Output Structure

- Single-pass results: `output/onepass/{model}/{dataset}_{config}_{run}.json`
- Two-pass results: `output/twopass/{model}/{dataset}_{config}_{run}.json`
- Per-model onepass summary: `output/{model}_onepass.json`
- Per-model two-pass summary: `output/{model}_twopass.json`
- Pre-computed token budget stats: `output/token_stats/{model_key}.json`
- Figures: `figures/`

## Code Style Notes

- Comments must be concise and fit on a single line

## Pipeline Overview

- One CLI entry point: `run`
- Two modes: `--onepass` (unconstrained, no token cap) and `--max-think-tokens N` (two-pass overflow)
- Two config profiles in `config/inference.yaml`: `greedy` (greedy decoding, temp=0, samples=1), `default` (model-recommended sampling, samples=3)
- Base models (with per-model default sampling params) registered in `config/models.yaml`
- HPC job submission via `scripts/submit_job.sh`; mode auto-derived from whether `max_think_tokens` is set

## Overflow Recovery Suffixes

Controlled via `--overflow-suffix` flag:

- `base`: nothing appended (pure truncation)
- `truncated`: appends " [reasoning truncated]"
- `formal`: appends "... I have to stop thinking and answer now."
- `human`: appends "... oops, I really need to stop thinking and to answer."

## Prompt Suffixes

Controlled via `--prompt-suffix` flag; **onepass only** (errors if combined with `--max-think-tokens`). Appends an instruction to the end of every user prompt to test whether prompt-engineering reduces reasoning length. Keys registered in `_PROMPT_SUFFIXES` in `src/run_inference.py`:

- `none`: nothing appended (baseline; keeps the standard `_onepass` filenames)
- `concise`: asks for brief reasoning and a guaranteed final answer
- `aspects`: replaces step-by-step reasoning with a short check of key problem aspects (inputs/outputs/edge cases/design choices) to force an early stop
- `plansolve`: PS+ plan-and-solve prompt (Wang et al., ACL 2023), adapted from arithmetic to code — a literature baseline that adds reasoning structure
- `budget`: TALE-EP token-budget trigger (Han et al., Findings of ACL 2025), fixed at an 8k token cap — a literature baseline that tells the model to limit its reasoning length

Non-baseline variants write to `output/onepass/{model}/{dataset}_{config}_mx{max_tokens}_onepass_{key}.json`.

## Data Format

- Inference datasets: `data/{type}/*.jsonl` — required key: `prompt`; type auto-detected from directory (`code`, `math`, `reasoning`, `crux`)
- Download all datasets by running `data/download.ipynb`

## External Dependencies

- Managed entirely with `uv`: declared in `pyproject.toml`, resolved into the committed `uv.lock`
- Use `uv add` / `uv remove` / `uv lock`; never hand-edit `uv.lock`, and never reintroduce a `requirements.txt`
- Runtime versions are pinned exactly so `uv sync` reproduces the experiment environment
- `llm_cgr` (llm-codegen-research): utility package — `save_json(data, path)`, `save_jsonl(data, path)`, `load_jsonl(file_path)`
