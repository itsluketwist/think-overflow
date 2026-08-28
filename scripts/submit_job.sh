#!/usr/bin/env bash

echo
echo ">>> Submitting inference jobs to slurm >>>"

# hardware constraints (adjust for your cluster)
GPU_CONSTRAINT="h200"
CPU_MEMORY="64G"

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
    # "0"

    # twopass reasoning caps (max_tokens = 32768)
    # 4096     # = 1/8
    # 8192     # = 2/8
    # 12288    # = 3/8
    # 16384    # = 4/8
    # 20480    # = 5/8
    # 24576    # = 6/8
    # 28672    # = 7/8
)

# overflow suffix keys (see _OVERFLOW_SUFFIXES in src/run_inference.py):
#   base      — nothing added (pure truncation)
#   dots      — appends "..."
#   truncated — appends " [reasoning truncated]"
#   formal    — appends "... I have to stop thinking and answer now."
#   human     — appends "... oops, I really need to stop thinking and to answer."
overflow_suffixes=(
    "base"
    # "dots"
    # "truncated"
    # "formal"
    # "human"
)

# prompt suffix keys (see _PROMPT_SUFFIXES in src/run_inference.py), onepass only:
#   none       — nothing added (baseline)
#   concise    — asks for brief reasoning and a guaranteed final answer
#   aspects    — swap step-by-step reasoning for a quick check of key problem aspects
#   plansolve  — PS+ plan-and-solve prompt (Wang et al., ACL 2023), adapted to code
#   budget     — TALE-EP token-budget trigger (Han et al., Findings of ACL 2025), fixed at 8k
prompt_suffixes=(
    # "none"    # baseline onepass (already run)
    "concise"
    "aspects"
    "plansolve"
    "budget"
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
echo "Prompt suffixes: ${prompt_suffixes[*]}"
echo "Update: $update"
echo

for model in "${models[@]}"; do
    for tokens in "${max_think_tokens[@]}"; do
        # mode: "" = onepass (prompt-suffix sweep), "0" = nothink, N = twopass (overflow sweep)
        if [ -z "$tokens" ]; then
            mode="onepass"
            think_flags=""
            suffixes=("${prompt_suffixes[@]}")
        elif [ "$tokens" = "0" ]; then
            mode="nothink"
            run_tag="nothink"
            think_flags="--max-think-tokens 0"
            suffixes=("")  # no overflow suffix for nothink — no reasoning to truncate
        else
            mode="twopass"
            suffixes=("${overflow_suffixes[@]}")
        fi

        for suffix in "${suffixes[@]}"; do
            # build human-readable run tag and mode-specific cli flags per suffix
            extra_flags=""
            if [ "$mode" = "onepass" ]; then
                # onepass sweeps prompt suffixes; "none" keeps the baseline run tag/flags
                if [ "$suffix" = "none" ]; then
                    run_tag="onepass"
                else
                    run_tag="onepass-${suffix}"
                    extra_flags="--prompt-suffix $suffix"
                fi
            elif [ "$mode" = "twopass" ]; then
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
    $extra_flags \
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
