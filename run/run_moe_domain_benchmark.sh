#!/bin/bash

# ======================================================================================
# Unified Script for MoE (Mixture of Experts) Model Domain Benchmark
#
# This script orchestrates the evaluation of MoE models on various domain-specific
# tasks. It combines and improves upon the functionalities of previous scripts
# (stat_moe_on_domains.sh, run_moe_stat.sh, run_dense.sh).
#
# Features:
# - Centralized configuration for easy modification of models, checkpoints, and domains.
# - Argument parsing for flags like --force-rerun.
# - Automated two-stage evaluation for each task:
#   1. Baseline run to gather general expert usage statistics.
#   2. Targeted run using domain-specific experts.
# - Skips already completed tasks to save time, unless --force-rerun is specified.
# - Submits jobs to the background for parallel execution.
# - Generates clean, separate log files for each run.
# ======================================================================================

# --- Configuration ---
MODEL_PATH="/aistor/sjtu/hpc_stor01/home/luoyijie/ckpts/fpm/Qwen2.5-MoE-5.7B-wiki-geography-common-32BS-func_tagged-router_loss"
CKPT_NAME="final"
DOMAINS=("geography" "history" "politics" "literature")

# Task definitions and their specific lm-eval arguments.
declare -A TASKS=(
    ["fpm_wiki"]="--num_fewshot 0 --batch_size 8"
    ["fpm_wiki_geography"]="--num_fewshot 0 --batch_size 8"
    ["fpm_wiki_history"]="--num_fewshot 0 --batch_size 8"
    ["fpm_wiki_literature"]="--num_fewshot 0 --batch_size 8"
    ["fpm_wiki_politics"]="--num_fewshot 0 --batch_size 8"
)

# --- Argument Parsing ---
FORCE_RERUN=false
for arg in "$@"; do
    case $arg in
        --force-rerun)
        FORCE_RERUN=true
        shift
        ;;
    esac
done

# --- Environment Setup ---
# Activate necessary environment for Ascend NPU
source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /usr/local/Ascend/nnal/atb/set_env.sh
export PATH=$PATH:~/.local/bin
export HCCL_CONNECT_TIMEOUT=3000
export HCCL_EXEC_TIMEOUT=3000
export HF_ALLOW_CODE_EVAL=1
export HF_DATASET_CACHE=/aistor/sjtu/hpc_stor01/home/luoyijie/data/lm_eval/cache
export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1

# --- Model and Path Preparation ---
MODEL_NAME_SUFFIX=${MODEL_PATH##*/}
CKPT_DIR=${MODEL_PATH}
if [ -n "$CKPT_NAME" ] && [ "$CKPT_NAME" != "final" ]; then
    CKPT_DIR="$MODEL_PATH/$CKPT_NAME"
    MODEL_NAME_SUFFIX="${MODEL_NAME_SUFFIX}_${CKPT_NAME}"
fi
OUTPUT_BASE="outputs/${MODEL_NAME_SUFFIX}"
mkdir -p logs

# --- Main Execution Loop ---
echo "Starting MoE domain benchmark for model: $MODEL_NAME_SUFFIX"
for task_name in "${!TASKS[@]}"; do
    eval_args="${TASKS[$task_name]}"

    # --- Run 1: Baseline statistics (no experts specified) ---
    OUTPUT_DIR_BASELINE="${OUTPUT_BASE}/stat_moe_baseline"
    LOG_FILE_BASELINE="logs/stat_${MODEL_NAME_SUFFIX}_${task_name}_baseline.log"
    
    if [ "$FORCE_RERUN" = true ] && compgen -G "${OUTPUT_DIR_BASELINE}/${task_name}/results_*.json" > /dev/null; then
        echo -e "\e[33m⚠️ Force rerun: deleting existing results for $task_name (baseline)\e[0m"
        rm -rf "${OUTPUT_DIR_BASELINE}/${task_name}"
    fi

    if compgen -G "${OUTPUT_DIR_BASELINE}/${task_name}/results_*.json" > /dev/null; then
        echo -e "\e[31m❌ Skipping completed task $task_name (baseline)\e[0m"
    else
        echo -e "\e[32m✅ Submitting baseline statistics job for ${task_name}\e[0m"
        CMD="accelerate launch -m lm_eval --model hf \
            --model_args pretrained=$CKPT_DIR,dtype=bfloat16 \
            --tasks $task_name \
            --include_path fpm_tasks \
            --log_samples \
            --show_config \
            --output_path ${OUTPUT_DIR_BASELINE}/${task_name} \
            --statistics_moe_experts \
            $eval_args"
        
        echo "   -> Log file: $LOG_FILE_BASELINE"
        eval "$CMD" > "$LOG_FILE_BASELINE" 2>&1 &
    fi

    # --- Run 2: Statistics with specified domain experts ---
    task_domain=""
    for domain in "${DOMAINS[@]}"; do
        if [[ "$task_name" == *"$domain"* ]]; then
            task_domain=$domain
            break
        fi
    done

    if [ -n "$task_domain" ]; then
        USE_EXPERTS_ARG="--use_experts $task_domain"
        OUTPUT_DIR_EXPERT="${OUTPUT_BASE}/stat_moe_use_${task_domain}"
        LOG_FILE_EXPERT="logs/stat_${MODEL_NAME_SUFFIX}_${task_name}_use_${task_domain}.log"

        if [ "$FORCE_RERUN" = true ] && compgen -G "${OUTPUT_DIR_EXPERT}/${task_name}/results_*.json" > /dev/null; then
            echo -e "\e[33m⚠️ Force rerun: deleting existing results for $task_name (expert: $task_domain)\e[0m"
            rm -rf "${OUTPUT_DIR_EXPERT}/${task_name}"
        fi

        if compgen -G "${OUTPUT_DIR_EXPERT}/${task_name}/results_*.json" > /dev/null; then
            echo -e "\e[31m❌ Skipping completed task $task_name (expert: $task_domain)\e[0m"
        else
            echo -e "\e[32m✅ Submitting statistics job for ${task_name} with expert '${task_domain}'\e[0m"
            CMD="accelerate launch -m lm_eval --model hf \
                --model_args pretrained=$CKPT_DIR,dtype=bfloat16 \
                --tasks $task_name \
                --include_path fpm_tasks \
                --log_samples \
                --show_config \
                --output_path ${OUTPUT_DIR_EXPERT}/${task_name} \
                --statistics_moe_experts \
                $USE_EXPERTS_ARG \
                $eval_args"

            echo "   -> Log file: $LOG_FILE_EXPERT"
            eval "$CMD" > "$LOG_FILE_EXPERT" 2>&1 
        fi
    fi
done

wait
echo -e "\n\e[1;34mAll benchmark jobs have been submitted and are running in the background.\e[0m"
echo "You can monitor their progress by checking the log files in the 'logs/' directory."
echo "Once all jobs are complete, results will be available in the '$OUTPUT_BASE' directory." 