# **think-overflow**

<div>
    <!-- badges from : https://shields.io/ -->
    <!-- logos available : https://simpleicons.org/ -->
    <a href="https://creativecommons.org/licenses/by/4.0/">
        <img alt="CC-BY-4.0 License" src="https://img.shields.io/badge/Licence-CC_BY_4.0-yellow?style=for-the-badge&logo=docs&logoColor=white" />
    </a>
    <a href="https://www.python.org/">
        <img alt="Python 3" src="https://img.shields.io/badge/Python_3-blue?style=for-the-badge&logo=python&logoColor=white" />
    </a>
</div>

## *about*

An empirical investigation into whether explicit reasoning helps **code generation** in small reasoning models.
We study *reasoning overflow* — the failure mode where a model exhausts its entire token budget on internal reasoning without producing an answer — and examine how accuracy changes across different levels of reasoning budget.
We evaluate 7 reasoning models across 6 code benchmarks (`EvalPlus`, `LiveCodeBench`, `BigCodeBench`, `CodeContests`, `CRUX-Input`,
`CRUX-Output`).

## *installation*

The code requires Python 3.11.6 to ensure valid reproduction of experiments.
Ensure you have it installed with the command below, otherwise download and install it from
[here](https://www.python.org/downloads/).

```shell
python --version
```

Now clone the repository code:

```shell
git clone https://github.com/itsluketwist/think-overflow
```

Once cloned, install the requirements locally in a virtual environment:

```shell
python3.11 -m venv .venv

source .venv/bin/activate

pip install -r requirements.frozen

pip install -e .
```

## *usage*

After [*installation*](#installation), use the `run` CLI to run inference. Two modes are available:

- **onepass** (omit `--max-think-tokens`) — single-pass unconstrained inference (baseline)
- **twopass** (`--max-think-tokens N`) — two-pass think-overflow inference with a reasoning cap

`--max-tokens` sets the overall token budget: total generation tokens for onepass, or combined
Pass 1 + Pass 2 tokens for twopass.

```shell
# onepass baseline — 32k budget, greedy decoding
run -m qwen3-8b -cp greedy --max-tokens 32768 -d code/evalplus,code/livecodebench,code/bigcodebench,code/code_contests,crux/cruxeval_i,crux/cruxeval_o

# two-pass think-overflow — 32k total budget, 8k reasoning cap, greedy decoding
run -m qwen3-8b -cp greedy --max-tokens 32768 --max-think-tokens 8192 --overflow-suffix formal -d code/evalplus,code/livecodebench,code/bigcodebench,code/code_contests,crux/cruxeval_i,crux/cruxeval_o
```

HPC job submission is handled via [`scripts/submit_job.sh`](scripts/submit_job.sh) — configure the
model list, datasets, and token caps there, then run the script to dispatch Slurm jobs.

### *debug*

Use `--debug` to smoke-test a run locally: it picks one dataset per eval type, limits each to
5 samples, and writes output to `output/debug/` instead of the normal results directories.

```shell
# debug onepass — greedy, 4k budget
run -m qwen3-0b -cp greedy --max-tokens 4096 -d code/evalplus,crux/cruxeval_i --debug

# debug two-pass — greedy, 8k total budget, 4k reasoning cap
run -m qwen3-0b -cp greedy --max-tokens 8192 --max-think-tokens 4096 --overflow-suffix formal -d code/evalplus,crux/cruxeval_i --debug
```

## *structure*

- [`config/`](config/) — inference profiles (`greedy`, `default`) and model registry
- [`data/`](data/) — benchmark datasets (`code/`, `crux/`, `math/`, `reasoning/`) and download notebook
- [`figures/`](figures/) — paper figures produced by the analysis notebooks
- [`harness/`](harness/) — code-execution evaluation harnesses (BigCodeBench, EditBench, CodeReval)
- [`notebooks/`](notebooks/) — analysis notebooks: `02_budget` (RQ1), `03_accuracy` (RQ2), `04_transition` (RQ3)
- [`output/`](output/) — inference results (`onepass/`, `twopass/`), per-model summaries, and token budget statistics
- [`scripts/`](scripts/) — HPC job submission (`submit_job.sh`) and token stats pre-computation (`compute_stats.py`)
- [`src/`](src/) — main package: CLI entry point, vLLM inference, evaluation, and utilities
- [`tests/`](tests/) — unit tests

## *development*

We use a few extra processes to ensure the code maintains a high quality.
First clone the project and create a virtual environment - as described above.

To update the frozen requirements after changing `requirements.txt`:

```shell
uv pip freeze > requirements.frozen
```

### *tests*

This project includes unit tests to ensure correct functionality.
Use [`pytest`](https://docs.pytest.org/en/stable/) to run the tests with:

```shell
pip install pytest

pytest tests
```

### *linting*

We use [`pre-commit`](https://pre-commit.com/) to lint the code, run it using:

```shell
pip install pre-commit

pre-commit run --all-files
```
