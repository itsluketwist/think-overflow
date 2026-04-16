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

After [*installation*](#installation), there are 2 ways to run the experiment code.
The easiest of which is via the the [`main.ipynb`](main.ipynb) notebook, which fully describes
each experiment and provides the methods to run them.

You can also use the `run` command from your terminal - this is likely best if you want to
reproduce the experiments on an external server or in a [docker](https://www.docker.com/)
container.

```shell
run --dataset-file data/example.json
```

All other non-experiment code that likely only needed to be ran a single time is explained in,
and can be interfaced with, via it's corresponding Jupyter notebook.
These notebooks are contained in the [`notebooks/`](notebooks/) directory, and are described in the
[*structure*](#structure) section.

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
