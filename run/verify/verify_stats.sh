#!/bin/bash

force_rerun=false
use_experts="history"
lm_eval_model="hf"
args=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --force-rerun)
            force_rerun=true
            shift
            ;;
        --use_experts)
            use_experts=$2
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


base=/aistor/sjtu/hpc_stor01/home/luoyijie/ckpts/fpm/Qwen2.5-MoE-5.7B-wiki-mean-common-32BS
ckpt_name=final
test_task=nq_open_history

echo "=== MOE Statistics Verification Script ==="
echo "Model: $base/$ckpt_name"
echo "Test task: $test_task"
echo "Use experts: $use_experts"
echo "Model type: $lm_eval_model"

source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /usr/local/Ascend/nnal/atb/set_env.sh
export PATH=$PATH:~/.local/bin
export HCCL_CONNECT_TIMEOUT=3000
export HCCL_EXEC_TIMEOUT=3000
export HF_ALLOW_CODE_EVAL=1
export HF_DATASET_CACHE=/aistor/sjtu/hpc_stor01/home/luoyijie/data/lm_eval/cache
export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1

modeling_patch=/aistor/sjtu/hpc_stor01/home/luoyijie/src/FPM/src/model/moe_monkey_patch.py
config_patch=/aistor/sjtu/hpc_stor01/home/luoyijie/src/FPM/src/model/config_patch.py

lm_eval_args="--num_fewshot 0 --batch_size 8 --limit 50"

model_name=${base##*/}
if [ -z "$ckpt_name" ]; then
    ckpt=$base
else
    ckpt=$base/$ckpt_name
    model_name=${model_name}_${ckpt_name}
fi

output_base=outputs/verify_stats/$model_name


    output_dir=$output_base/hf/use_$use_experts
    launch="accelerate launch -m lm_eval --model hf"
    model_args="pretrained=$ckpt,dtype=bfloat16,path_to_modeling_monkey_patch=$modeling_patch,path_to_config_monkey_patch=$config_patch"


mkdir -p "$output_dir"
echo "Output directory: $output_dir"


echo -e "\e[32m🚀\e[0m  Running verification for ${model_name}_${test_task} with experts: $use_experts"

cmd="$launch \
    --model_args \"$model_args\" \
    --tasks \"$test_task\" \
    --include_path /aistor/sjtu/hpc_stor01/home/luoyijie/src/lm-evaluation-harness/fpm_tasks \
    --log_samples \
    --show_config \
    --output_path \"$output_dir/$test_task\" \
    --statistics_moe_experts \
    --use_experts $use_experts \
    $lm_eval_args"

echo "Command to execute:"
echo "$cmd"
echo ""

eval "$cmd"

echo ""
echo "=== Verification Results ==="
if compgen -G "$output_dir/$test_task/**/stats_moe_*.json" > /dev/null; then
    echo -e "\e[32m✅\e[0m  Stats files generated successfully!"
    echo "Generated stats files:"
    ls -la "$output_dir/$test_task"/**/stats_moe_*.json
    
    echo ""
    echo "Stats file content preview:"
    stats_file=$(ls "$output_dir/$test_task"/**/stats_moe_*.json | head -1)
    if [ -f "$stats_file" ]; then
        echo "File: $stats_file"
        head -20 "$stats_file"
    fi
else
    echo -e "\e[31m❌\e[0m  No stats files found! Verification failed."
    exit 1
fi

