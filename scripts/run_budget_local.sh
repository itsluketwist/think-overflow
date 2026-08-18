#!/usr/bin/env bash
set -euo pipefail

# runs the "budget" prompt-suffix onepass sweep directly on this machine (no slurm) —
# for use on a rented gpu box rather than the hpc cluster. mirrors submit_job.sh but
# executes each `run` command in sequence instead of submitting sbatch jobs.

echo
echo ">>> Running budget prompt-suffix jobs locally >>>"

###############################################################################
#                              model configuration                            #
###############################################################################

models=(
    "qwen3-8b"
    "qwen3-14b"
    "qwen3-32b"

    "olmo-3-7b"
    "llama-r1-8b"
    "or-7b"
    "ocr-7b"
)

###############################################################################
#                           inference configuration                           #
###############################################################################

# datasets to evaluate
eval_datasets=(
    "code/evalplus"
    "code/livecodebench"
    "code/bigcodebench"
    "code/code_contests"
    "crux/cruxeval_i"
    "crux/cruxeval_o"
)

# config profile for inference (from config/inference.yaml)
inference_config="greedy"
# inference_config="default"

# overall token budget (onepass, so this is the full generation budget)
max_tokens=32768

# force regeneration even if output file already exists
update=false

###############################################################################
#                              environment setup                              #
###############################################################################

echo
echo "Loading virtual environment..."
source .venv/bin/activate
echo

echo "Loading environment variables from .env..."
if [ -f .env ]; then
    source .env
else
    echo "WARNING: .env file not found! Create one with HF_HOME, HF_TOKEN, etc."
fi
echo

# use expandable memory segments to prevent pytorch cuda allocator fragmentation,
# which causes oom errors on long-sequence training even when total memory is available
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# bigcodebench evaluation requires a dedicated venv with pinned library versions
export BCB_PYTHON="$(pwd)/harness/.venv/bin/python"

###############################################################################
#                                run jobs                                     #
###############################################################################

datasets_csv=$(IFS=,; echo "${eval_datasets[*]}")

update_flag=""
if [ "$update" = true ]; then
    update_flag="--update"
fi

mkdir -p logs

echo
echo "Models: ${#models[@]}"
echo "Eval datasets: ${#eval_datasets[@]}"
echo "  $datasets_csv"
echo "Config: $inference_config"
echo "Max tokens: $max_tokens"
echo "Prompt suffix: budget"
echo "Update: $update"
echo

for model in "${models[@]}"; do
    log_file="logs/infer-${model}-mx${max_tokens}-onepass-budget.out"

    echo "Running: $model | mx=$max_tokens | onepass-budget"
    echo "  logging to $log_file"

    run \
        -m "$model" \
        -d "$datasets_csv" \
        --config-profile "$inference_config" \
        --max-tokens "$max_tokens" \
        --prompt-suffix budget \
        $update_flag \
        2>&1 | tee "$log_file"

    echo "Finished: $model | mx=$max_tokens | onepass-budget"
    echo
done

echo
echo "All budget runs complete!"
echo
