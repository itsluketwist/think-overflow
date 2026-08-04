# **BigCodeBench Evaluation Harness**

Direct Python evaluation of BigCodeBench using the official `bigcodebench` environment.

## *why a separate environment?*

BigCodeBench test code imports many packages (numpy, pandas, scipy, PIL, sklearn, etc.)
at specific versions pinned by the benchmark authors. Installing these alongside the
main project dependencies would cause conflicts, so a dedicated virtual environment is
used here.

## *setup*

```bash
# go to the harness directory
cd harness

# create the correct venv
uv venv .venv --python 3.10.20

# install harness requirements
uv pip install --python .venv/bin/python -r requirements.harness
```
