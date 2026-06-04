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

# overall token budget (pass 1 + pass 2 combined for twopass, full generation for onepass)
# max_tokens=null
# max_tokens=8192
# max_tokens=16384
max_tokens=32768

# reasoning token caps to sweep (pass 1 max tokens); "" means onepass (no cap)
max_think_tokens=(
    # onepass - baseline result with reasoning
    ""

    # nothink - run with reasoning disabled
    "0"

    # max_tokens = 32768
    4096     # = 1/8
    8192     # = 2/8
    12288    # = 3/8
    16384    # = 4/8
    20480    # = 5/8
    24576    # = 6/8
    28672    # = 7/8
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

# force regeneration even if output file already exists
update=false

###############################################################################
#                              submit jobs                                    #
###############################################################################

datasets_csv=$(IFS=,; echo "${eval_datasets[*]}")

update_flag=""
if [ "$update" = true ]; then
    update_flag="--update"
fi

echo
echo "Models: ${#models[@]}"
echo "Eval datasets: ${#eval_datasets[@]}"
echo "  $datasets_csv"
echo "Config: $inference_config"
echo "Max tokens: $max_tokens"
echo "Max think tokens: ${max_think_tokens[*]}"
echo "Overflow suffixes: ${overflow_suffixes[*]}"
echo "Update: $update"
echo

for model in "${models[@]}"; do
    for tokens in "${max_think_tokens[@]}"; do
        # determine mode: "" = onepass, "0" = nothink, N = twopass sweep over suffixes
        if [ -z "$tokens" ]; then
            run_tag="onepass"
            think_flags=""
            suffixes=("")
        elif [ "$tokens" = "0" ]; then
            run_tag="nothink"
            think_flags="--max-think-tokens 0"
            suffixes=("")  # no overflow suffix for nothink — no reasoning to truncate
        else
            suffixes=("${overflow_suffixes[@]}")
        fi

        for suffix in "${suffixes[@]}"; do
            # build human-readable run tag and optional cli flags for twopass
            if [ "$tokens" != "" ] && [ "$tokens" != "0" ]; then
                run_tag="th${tokens}-${suffix}"
                think_flags="--max-think-tokens $tokens --overflow-suffix $suffix"
            fi

            echo "Submitting: $model | mx=$max_tokens | $run_tag"

            sbatch <<EOF
#!/bin/bash -l
#SBATCH --job-name=infer-${model}-mx${max_tokens}-${run_tag}
#SBATCH --output=/users/%u/code/think-overflow/logs/infer-${model}/mx${max_tokens}-${run_tag}-%j.out
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
    --max-tokens $max_tokens \
    $think_flags \
    $update_flag

echo "Ending job: $model | mx=$max_tokens | $run_tag"
EOF

            # small delay between submissions to avoid scheduler overload
            sleep 0.5
        done
    done
done

echo
echo "All jobs submitted!"
echo
