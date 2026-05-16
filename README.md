# **research-template**

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

todo

## *installation*

The code requires Python 3.11.6 to ensure valid reproduction of experiments.
Ensure you have it installed with the command below, otherwise download and install it from
[here](https://www.python.org/downloads/).

```shell
python --version
```

Now clone the repository code:

```shell
git clone **redacted**
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
run -m qwen3-8b -cp greedymax --max-tokens 32768 -d code/evalplus,math/gsm8k,reasoning/gpqa,crux/cruxeval_i

# two-pass think-overflow — 32k total budget, 8k reasoning cap, greedy decoding
run -m qwen3-8b -cp greedy --max-tokens 32768 --max-think-tokens 8192 --overflow-suffix formal -d code/evalplus,math/gsm8k,reasoning/gpqa,crux/cruxeval_i
```

HPC job submission is handled via [`scripts/submit_job.sh`](scripts/submit_job.sh) — configure the
model list, datasets, and token caps there, then run the script to dispatch Slurm jobs.

### *debug*

Use `--debug` to smoke-test a run locally: it picks one dataset per eval type, limits each to
5 samples, and writes output to `output/debug/` instead of the normal results directories.

```shell
# debug onepass — greedy, 32k budget
run -m qwen3-0b -cp greedy --max-tokens 4096 -d code/evalplus,math/gsm8k,reasoning/gpqa,crux/cruxeval_i --debug

# debug two-pass — greedy, 32k total budget, 8k reasoning cap
run -m qwen3-0b -cp creative --max-tokens 32768 --max-think-tokens 8192 --overflow-suffix formal -d code/evalplus,math/gsm8k,reasoning/gpqa,crux/cruxeval_i --debug
```

## *structure*

todo

- [`data/`](data/) - The data used in the project.
- [`output/`](output/) - The generated results.
- [`src/`](src/) - The main project code.

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
