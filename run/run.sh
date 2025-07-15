#!/bin/bash

force_rerun=false
task_type=""
lm_eval_model="hf"
use_experts_arg=""
use_experts_name=""
args=()

# Parse arguments
while [[ $# -gt 0 ]]; do
    case "$1" in
        --task)
            task_type=$2
            shift 2
            ;;
        --force-rerun)
            force_rerun=true
            shift
            ;;
        --use_experts)
            use_experts_name=$2
            use_experts_arg="--use_experts $2"
            shift 2
            ;;
        --model)
            lm_eval_model=$2
            shift 2
            ;;
        *)
            args+=("$1")
            shift
            ;;
    esac
done
set -- "${args[@]}"

# Usage check
if [ "$task_type" != "dense" ] && [ "$task_type" != "moe" ] && [ "$task_type" != "moe_stat" ]; then
    echo "Usage: $0 --task <dense|moe|moe_stat> [options] <ckpt_base_dir> <ckpt_name> [domain1 domain2...]"
    echo ""
    echo "Options:"
    echo "  --force-rerun           Force rerun existing results"
    echo "  --use_experts <value>   Specify experts to use"
    echo "  --model <hf|vllm>       Override evaluation model (default: hf for moe, predefined for dense)"
    echo ""
    echo "Task modes:"
    echo "  dense:    For dense models"
    echo "            - No domains: test baseline tasks (fpm_wiki_main_topics, etc.)"
    echo "            - With domains: test specific domain tasks (fpm_wiki_<domain>)"
    echo "  moe:      For MoE models, requires domain specification"
    echo "  moe_stat: For MoE statistics, requires domain specification"
    echo ""
    echo "Examples:"
    echo "  $0 --task dense /path/ckpt name                    # Test baseline tasks"
    echo "  $0 --task dense /path/ckpt name culture history    # Test specific domains"
    echo "  $0 --task moe /path/ckpt name culture entertainment # Test MoE domains"
    exit 1
fi

if [ $# -lt 2 ]; then
    echo "Error: Missing required arguments <ckpt_base_dir> and <ckpt_name>"
    echo "Run '$0 --task invalid' for usage information"
    exit 1
fi

base=$1
ckpt_name=$2
shift 2
domains=("$@")

# Environment setup
setup_env() {
    source /usr/local/Ascend/ascend-toolkit/set_env.sh
    source /usr/local/Ascend/nnal/atb/set_env.sh
    export PATH=$PATH:~/.local/bin
    export HCCL_CONNECT_TIMEOUT=3000
    export HCCL_EXEC_TIMEOUT=3000
    export HF_ALLOW_CODE_EVAL=1
    export HF_DATASET_CACHE=/aistor/sjtu/hpc_stor01/home/luoyijie/data/lm_eval/cache
    export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
    export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
    export ASCEND_WORK_PATH=/aistor/sjtu/hpc_stor01/home/luoyijie/src/lm-evaluation-harness/logs/ascend-log

    # Patches for different task types
    modeling_patch=/aistor/sjtu/hpc_stor01/home/luoyijie/src/FPM/src/model/moe_monkey_patch.py
    config_patch=/aistor/sjtu/hpc_stor01/home/luoyijie/src/FPM/src/model/config_patch.py
    fpm_vllm_patch=/aistor/sjtu/hpc_stor01/home/luoyijie/src/FPM/fpm_vllm_patch.py

    # npu-smi info
    # pgrep -af python
}

# Task configurations - All available tasks are explicitly listed
# Dense baseline tasks (for general/baseline models, no domain specified)
declare -A dense_tasks=(
    ["fpm_wiki_main_topics"]="vllm#--num_fewshot 0"
)

# Dense domain tasks (for domain-specific models, use dense config without MoE patches)
declare -A dense_domain_tasks=(
    ["fpm_wiki_culture"]="vllm#--num_fewshot 0"
    ["fpm_wiki_entertainment"]="vllm#--num_fewshot 0"
    ["fpm_wiki_geography"]="vllm#--num_fewshot 0"
    ["fpm_wiki_history"]="vllm#--num_fewshot 0"
    ["fpm_wiki_humanities"]="vllm#--num_fewshot 0"
    ["fpm_wiki_science"]="vllm#--num_fewshot 0"
)

# MoE tasks (for MoE models, use MoE config with patches)
declare -A moe_tasks=(
    ["fpm_wiki_culture"]="hf#--num_fewshot 0 --batch_size 32"
    ["fpm_wiki_entertainment"]="hf#--num_fewshot 0 --batch_size 32"
    ["fpm_wiki_geography"]="hf#--num_fewshot 0 --batch_size 32"
    ["fpm_wiki_history"]="hf#--num_fewshot 0 --batch_size 32"
    ["fpm_wiki_humanities"]="hf#--num_fewshot 0 --batch_size 32"
    ["fpm_wiki_science"]="hf#--num_fewshot 0 --batch_size 32"
    # Note: MoE stat mode also uses extra_moe_stats_tasks
)

# Extra MoE stats tasks (only for MoE stat mode)
declare -A extra_moe_stats_tasks=(
    ["stats_pretrain_nq_open_geography"]="hf#--batch_size 16"
    ["stats_pretrain_nq_open_history"]="hf#--batch_size 16"
    ["stats_pretrain_nq_open_literature"]="hf#--batch_size 16"
    ["stats_pretrain_nq_open_politics"]="hf#--batch_size 16"
)

# Build model path
model_name=${base##*/}
if [ -z "$ckpt_name" ]; then
    ckpt=$base
else
    ckpt=$base/$ckpt_name
    model_name=${model_name}_${ckpt_name}
fi
output_base=outputs/$model_name

# Helper functions to reduce code duplication

# Build output directory path
build_output_dir() {
    local eval_model=$1
    local base_dir
    
    if [ "$eval_model" == "vllm" ]; then
        base_dir=$output_base/vllm
    elif [ "$eval_model" == "hf" ]; then
        base_dir=$output_base/hf
    fi
    
    if [ "$use_experts_arg" != "" ]; then
        base_dir=$base_dir/use_$use_experts_name
    fi
    
    echo "$base_dir"
}

# Check for existing results and handle force rerun
check_existing_results() {
    local task_name=$1
    local output_dir=$2
    local result_pattern="results_*.json"
    
    if [ "$task_type" == "moe_stat" ]; then
        result_pattern="stats_moe_*.json"
    fi
    
    if compgen -G "$output_dir/$task_name/**/$result_pattern" > /dev/null; then
        if [ "$force_rerun" = true ]; then
            echo -e "\e[33m⚠️\e[0m Force rerun: deleting existing results for $task_name under $output_dir/$task_name"
            rm -rf "$output_dir/$task_name"
            return 0  # Continue with execution
        else
            echo -e "\e[31m❌\e[0m Skipping completed task $task_name, existing: $output_dir/$task_name"
            return 1  # Skip this task
        fi
    fi
    return 0  # No existing results, continue
}

# Generate evaluation command
generate() {
    local task_name=$1
    local eval_model=$2
    local eval_args=$3
    local output_dir=$4
    local extra_args=""
    
    if [ "$task_type" == "moe_stat" ]; then
        extra_args="--statistics_moe_experts"
    fi

    if [ "$eval_model" == "vllm" ]; then
        launch="lm_eval --model vllm"
        if [ "$task_type" == "dense" ]; then
            model_args="pretrained=$ckpt,dtype=bfloat16,max_model_len=8192,tensor_parallel_size=1,gpu_memory_utilization=0.95,data_parallel_size=8"
        else
            model_args="pretrained=$ckpt,dtype=bfloat16,max_model_len=8192,tensor_parallel_size=1,gpu_memory_utilization=0.95,data_parallel_size=8,path_to_fpm_vllm_patch=$fpm_vllm_patch"
            if [ ! -z "$use_experts_name" ]; then
                model_args="$model_args,use_experts=$use_experts_name"
            fi
        fi
    elif [ "$eval_model" == "hf" ]; then
        launch="accelerate launch -m lm_eval --model hf"
        if [ "$task_type" == "dense" ]; then
            model_args="pretrained=$ckpt,dtype=auto"
        else
            model_args="pretrained=$ckpt,dtype=bfloat16,path_to_modeling_monkey_patch=$modeling_patch,path_to_config_monkey_patch=$config_patch"
        fi
    else
        return 1
    fi

    echo "$launch" \
        --model_args "$model_args" \
        --tasks "$task_name" \
        --include_path /aistor/sjtu/hpc_stor01/home/luoyijie/src/lm-evaluation-harness/fpm_tasks \
        --log_samples \
        --show_config \
        --output_path "$output_dir/$task_name" \
        "$eval_args $use_experts_arg $extra_args"
}

# Run a single task with all common logic
run_task() {
    local task_name=$1
    local eval_model=$2
    local eval_args=$3
    
    # Build output directory
    local output_dir=$(build_output_dir "$eval_model")
    mkdir -p "$output_dir"
    
    # Check existing results
    if ! check_existing_results "$task_name" "$output_dir"; then
        return 0  # Skip this task
    fi

    setup_env
    
    # Execute the task
    echo -e "\e[32m✅\e[0m Submitting ${model_name}_${task_name}, output_dir: $output_dir"
    local cmd=$(generate "$task_name" "$eval_model" "$eval_args" "$output_dir")
    echo "$cmd"
    eval "$cmd"
}

# Main execution loop
if [ "$task_type" == "dense" ]; then
    # Dense mode: two usage patterns
    if [ ${#domains[@]} -eq 0 ]; then
        # Pattern 1: No domains specified - use predefined tasks for baseline/general models
        for task_name in "${!dense_tasks[@]}"; do
            IFS="#" read -r eval_model eval_args <<< "${dense_tasks[$task_name]}"
            run_task "$task_name" "$eval_model" "$eval_args"
        done
    else
        # Pattern 2: Domains specified - test specific fpm_wiki_<domain> tasks for domain models
        for domain in "${domains[@]}"; do
            # Support both direct task names and domain prefix
            if [[ "$domain" == fpm_wiki_* ]]; then
                task_name="$domain"
            else
                task_name="fpm_wiki_$domain"
            fi
            
            # Check if task exists in dense_domain_tasks (domain tasks with dense config)
            if [[ ! -v dense_domain_tasks["$task_name"] ]]; then
                echo "Warning: Task $task_name not found in dense domain configuration, skipping"
                continue
            fi
            
            IFS="#" read -r eval_model eval_args <<< "${dense_domain_tasks[$task_name]}"
            
            # Override model if specified via command line
            if [ "$lm_eval_model" != "hf" ]; then
                eval_model="$lm_eval_model"
            fi
            
            run_task "$task_name" "$eval_model" "$eval_args"
        done
    fi
else
    # MoE/MoE_stat mode: use domains from command line
    if [ ${#domains[@]} -eq 0 ]; then
        echo "Error: At least one domain must be specified for $task_type mode"
        exit 1
    fi
    
    for domain in "${domains[@]}"; do
        task_name=""
        eval_model=""
        eval_args=""
        
        # Try to find task in moe_tasks first
        if [[ "$domain" == fpm_wiki_* ]]; then
            task_name="$domain"
        else
            task_name="fpm_wiki_$domain"
        fi
        
        if [[ -v moe_tasks["$task_name"] ]]; then
            IFS="#" read -r eval_model eval_args <<< "${moe_tasks[$task_name]}"
        elif [ "$task_type" == "moe_stat" ] && [[ -v extra_moe_stats_tasks["$domain"] ]]; then
            # For moe_stat mode, also check extra_moe_stats_tasks with exact domain name
            task_name="$domain"
            IFS="#" read -r eval_model eval_args <<< "${extra_moe_stats_tasks[$task_name]}"
        else
            echo "Warning: Task $task_name not found in configuration, skipping"
            continue
        fi
        
        # Override model if specified via command line
        if [ "$lm_eval_model" != "hf" ]; then
            eval_model="$lm_eval_model"
        fi
        
        run_task "$task_name" "$eval_model" "$eval_args"
    done
fi