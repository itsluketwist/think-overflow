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

Dependencies are managed with [`uv`](https://docs.astral.sh/uv/), which is the only tool you need
installed up front — it fetches the correct Python version (3.11.6, as used for all reported
experiments) for you:

```shell
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Now clone the repository code:

```shell
git clone https://github.com/itsluketwist/think-overflow
```

Once cloned, create the virtual environment and install the exact locked dependencies:

```shell
cd think-overflow

uv sync --locked
```

This creates `.venv/` and installs the project itself in editable mode, so the `run` CLI is
available. Either activate the environment, or prefix commands with `uv run`:

```shell
source .venv/bin/activate

# ...or, without activating:
uv run run --help
```

BigCodeBench evaluation runs its test code in a second, dedicated venv, since the benchmark pins
library versions that conflict with the ones above — see [`harness/README.md`](harness/README.md)
to create it before running any `code/bigcodebench` evaluation.

## *usage*

After [*installation*](#installation), use the `run` CLI to run inference. Two modes are available:

- **onepass** (omit `--max-think-tokens`) — single-pass unconstrained inference (baseline)
- **twopass** (`--max-think-tokens N`) — two-pass think-overflow inference with a reasoning cap, following
  the budget-forcing approach of [Muennighoff et al. (s1, EMNLP 2025)](https://aclanthology.org/2025.emnlp-main.1025/)

`--max-tokens` sets the overall token budget: total generation tokens for onepass, or combined
Pass 1 + Pass 2 tokens for twopass.

```shell
# onepass baseline — 32k budget, greedy decoding
run -m qwen3-8b -cp greedy --max-tokens 32768 -d code/evalplus,code/livecodebench,code/bigcodebench,code/code_contests,crux/cruxeval_i,crux/cruxeval_o

# two-pass think-overflow — 32k total budget, 8k reasoning cap, greedy decoding
run -m qwen3-8b -cp greedy --max-tokens 32768 --max-think-tokens 8192 --overflow-suffix formal -d code/evalplus,code/livecodebench,code/bigcodebench,code/code_contests,crux/cruxeval_i,crux/cruxeval_o
```

For onepass runs, `--prompt-suffix` appends an instruction to every prompt to test whether prompt-engineering
shortens reasoning (keys registered in `_PROMPT_SUFFIXES` in [`src/run_inference.py`](src/run_inference.py); `none`
is the baseline). It is onepass-only and errors if combined with `--max-think-tokens`. The `plansolve` key adapts
the PS+ plan-and-solve prompt from [Wang et al. (ACL 2023)](https://aclanthology.org/2023.acl-long.147/).

```shell
# onepass with a prompt-engineering suffix — writes to a distinct results file
run -m qwen3-8b -cp greedy --max-tokens 32768 --prompt-suffix concise -d code/evalplus,crux/cruxeval_i
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
- [`harness/`](harness/) — dedicated venv used to execute BigCodeBench test code
- [`notebooks/`](notebooks/) — analysis notebooks: `02_budget` (RQ1), `03_accuracy` (RQ2), `04_transition` (RQ3)
- [`output/`](output/) — inference results (`onepass/`, `twopass/`), per-model summaries, and token budget statistics
- [`scripts/`](scripts/) — HPC job submission (`submit_job.sh`) and token stats pre-computation (`compute_stats.py`)
- [`src/`](src/) — main package: CLI entry point, vLLM inference, evaluation, and utilities
- [`tests/`](tests/) — unit tests

## *development*

We use a few extra processes to ensure the code maintains a high quality.
First clone the project and create a virtual environment - as described above.

### *dependencies*

Dependencies live in the `dependencies` list of [`pyproject.toml`](pyproject.toml), and the exact
resolved versions of every direct and transitive dependency are recorded in `uv.lock` (which is
committed, and should never be edited by hand).

```shell
# add or remove a dependency, updating pyproject.toml and uv.lock together
uv add <package>
uv remove <package>

# re-resolve uv.lock after editing pyproject.toml by hand
uv lock

# upgrade every dependency to the newest versions the constraints allow
uv lock --upgrade

# check uv.lock is in sync with pyproject.toml, without changing anything
uv lock --check
```

Note that the runtime dependencies are deliberately pinned to exact versions, so that the reported
experiments can be reproduced — upgrade them only intentionally.

### *tests*

This project includes unit tests to ensure correct functionality.
Use [`pytest`](https://docs.pytest.org/en/stable/) to run the tests with:

```shell
uv run pytest tests
```

### *linting*

We use [`pre-commit`](https://pre-commit.com/) to lint the code, run it using:

```shell
uv run pre-commit run --all-files
```
