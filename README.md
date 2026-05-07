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

- **`--onepass`** — single-pass unconstrained inference (baseline)
- **`--max-think-tokens N`** — two-pass think-overflow inference with a reasoning cap

```shell
# onepass baseline — full budget, greedy decoding
run -m qwen3-8b -cp greedymax -d code/evalplus,math/gsm8k,reasoning/gpqa,crux/cruxeval_i --onepass

# two-pass think-overflow — 4096 reasoning token cap, greedy decoding
run -m qwen3-8b -cp greedy -d code/evalplus,math/gsm8k,reasoning/gpqa,crux/cruxeval_i --max-think-tokens 4096 --overflow-suffix formal
```

HPC job submission is handled via [`scripts/submit_job.sh`](scripts/submit_job.sh) — configure the
model list, datasets, and token caps there, then run the script to dispatch Slurm jobs.

### *debug*

Use `--debug` to smoke-test a run locally: it picks one dataset per eval type, limits each to
5 samples, and writes output to `output/debug/` instead of the normal results directories.

```shell
# debug onepass — model-default sampling, 3 samples per task
run -m qwen3-1.7b -cp default -d code/evalplus,math/gsm8k,reasoning/gpqa,crux/cruxeval_i --onepass --debug

# debug two-pass — greedy, 4096 token reasoning cap
run -m qwen3-1.7b -cp greedy -d code/evalplus,math/gsm8k,reasoning/gpqa,crux/cruxeval_i --max-think-tokens 4096 --overflow-suffix formal --debug
```

## *structure*

todo

- [`data/`](data/) - The data used in the project.
- [`output/`](output/) - The generated results.
- [`src/`](src/) - The main project code.

## *development*

We use a few extra processes to ensure the code maintains a high quality.
First clone the project and create a virtual environment - as described above.

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
