#!/usr/bin/env bash

echo
echo ">>> Submitting inference jobs to slurm >>>"

# hardware constraints (adjust for your cluster)
GPU_CONSTRAINT="h200"
CPU_MEMORY="96G"

###############################################################################
#                              model configuration                            #
###############################################################################

models=(
    "qwen3-8b"
    "olmo-3-7b-think"
    "code-nemotron-7b"
)

###############################################################################
#                           inference configuration                           #
###############################################################################

# datasets to evaluate
eval_datasets=(
    "code/evalplus"
    "code/livecodebench"
    "code/bigcodebench"
    "code/editbench"
    "crux/cruxeval_i"
    "crux/cruxeval_o"
    "reasoning/gpqa"
    "math/gsm8k"
)

# config profile for inference (from config/inference.yaml)
inference_config="greedy"

# reasoning token caps to sweep (pass 1 max tokens)
max_think_tokens=(
    # 4096
    # 8192
    # 12288
    # 16384
    # 20480
    # 24576
    # 28664
)

# overflow suffix keys (see _OVERFLOW_SUFFIXES in src/run_infer.py):
#   base      — nothing added (pure truncation)
#   truncated — appends " [reasoning truncated]"
#   formal    — appends "... I have to stop thinking and answer now."
#   human     — appends "... oops, I really need to stop thinking and to answer."
overflow_suffixes=(
    "base"
    # "blank"
    # "truncated"
    # "formal"
    # "human"
)

# run single-pass baseline inference instead of two-pass inference
baseline=true

# force regeneration even if output file already exists
update=true

###############################################################################
#                              submit jobs                                    #
###############################################################################

datasets_csv=$(IFS=,; echo "${eval_datasets[*]}")

baseline_flag=""
if [ "$baseline" = true ]; then
    baseline_flag="--baseline"
fi

update_flag=""
if [ "$update" = true ]; then
    update_flag="--update"
fi

echo
echo "Models: ${#models[@]}"
echo "Eval datasets: ${#eval_datasets[@]}"
echo "  $datasets_csv"
echo "Config: $inference_config"
echo "Baseline: $baseline"
if [ "$baseline" = false ]; then
    echo "Max think tokens: ${max_think_tokens[*]}"
    echo "Overflow suffixes: ${overflow_suffixes[*]}"
fi
echo "Update: $update"
echo

if [ "$baseline" = true ]; then
    # submit one job per model for a single-pass baseline run
    for model in "${models[@]}"; do
        echo "Submitting: $model | baseline"

        sbatch <<EOF
#!/bin/bash -l
#SBATCH --job-name=infer-${model}-baseline
#SBATCH --output=/users/%u/logs/infer-${model}-baseline/%j.out
#SBATCH --partition=gpu
#SBATCH --gres=gpu
#SBATCH --gpus=1
#SBATCH --mem=${CPU_MEMORY}
#SBATCH --constraint=${GPU_CONSTRAINT}

source ./scripts/setup_job.sh

run \
    -m $model \
    -d $datasets_csv \
    --config-profile $inference_config \
    --baseline \
    $update_flag

echo "Ending job: $model | baseline"
EOF

        # small delay between submissions to avoid scheduler overload
        sleep 0.5
    done
else
    # submit one job per model × max_think_tokens × overflow_suffix combination
    for model in "${models[@]}"; do
        for tokens in "${max_think_tokens[@]}"; do
            for suffix in "${overflow_suffixes[@]}"; do
                echo "Submitting: $model | mt=$tokens | suffix=$suffix"

                sbatch <<EOF
#!/bin/bash -l
#SBATCH --job-name=infer-${model}-mt${tokens}-${suffix}
#SBATCH --output=/users/%u/logs/infer-${model}-mt${tokens}-${suffix}/%j.out
#SBATCH --partition=gpu
#SBATCH --gres=gpu
#SBATCH --gpus=1
#SBATCH --mem=${CPU_MEMORY}
#SBATCH --constraint=${GPU_CONSTRAINT}

source ./scripts/setup_job.sh

run \
    -m $model \
    -d $datasets_csv \
    --config-profile $inference_config \
    --max-think-tokens $tokens \
    --overflow-suffix $suffix \
    $update_flag

echo "Ending job: $model | mt=$tokens | suffix=$suffix"
EOF

                # small delay between submissions to avoid scheduler overload
                sleep 0.5
            done
        done
    done
fi

echo
echo "All jobs submitted!"
echo
