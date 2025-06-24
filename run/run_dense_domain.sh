#!/bin/bash

force_rerun=false
lm_eval_model="hf"
args=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --force-rerun)
            force_rerun=true
            shift
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

if [ $# -lt 3 ]; then
    echo "Usage: $0 [--force-rerun] [--model <hf|vllm>] <ckpt_base_dir> <ckpt_name> <domain1> [domain2...]"
    exit 1
fi
base=$1; shift
ckpt_name=$1; shift
domains=("$@")

# interactive scripts have to do it first
source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /usr/local/Ascend/nnal/atb/set_env.sh
export PATH=$PATH:~/.local/bin
export HCCL_CONNECT_TIMEOUT=3000
export HCCL_EXEC_TIMEOUT=3000
export HF_ALLOW_CODE_EVAL=1
export HF_DATASET_CACHE=/aistor/sjtu/hpc_stor01/home/luoyijie/data/lm_eval/cache
export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1

lm_eval_args="--num_fewshot 0 --batch_size 32"

# build env for model
model_name=${base##*/}
if [ -z "$ckpt_name" ]; then
    # no intermediate ckpts, or subdirs, use directly base dir as ckpt dir
    ckpt=$base
else
    ckpt=$base/$ckpt_name
    model_name=${model_name}_${ckpt_name}
fi
output_base=outputs/$model_name

# generate eval cmd
# task_name="$1"; lm_eval_model="$2"; lm_eval_args="$3";"
generate() { 

    if [ "$2" == "vllm" ]; then
        launch="lm_eval --model vllm"
        model_args="pretrained=$ckpt,dtype=bfloat16,max_model_len=4096,tensor_parallel_size=1,gpu_memory_utilization=0.95,data_parallel_size=8"
    elif [ "$2" == "hf" ]; then
        launch="accelerate launch -m lm_eval --model hf"
        model_args="pretrained=$ckpt,dtype=bfloat16"
    else
        return 1
    fi

    echo "$launch" \
        --model_args "$model_args" \
        --tasks "$1" \
        --include_path /aistor/sjtu/hpc_stor01/home/luoyijie/src/lm-evaluation-harness/fpm_tasks \
        --log_samples \
        --show_config \
        --output_path "$output_dir/$1" \
        "$3"
}

# prepare-and-submit loop
for task_name in "${domains[@]}"; do

    if [ "$lm_eval_model" == "vllm" ]; then
        output_dir=$output_base/vllm
    elif [ "$lm_eval_model" == "hf" ]; then
        output_dir=$output_base/hf
    fi

    mkdir -p "$output_dir"
    echo "output_dir: $output_dir"

    if compgen -G "$output_dir/$task_name/**/results_*.json" > /dev/null; then
        if [ "$force_rerun" = true ]; then
            echo -e "\e[33m⚠️\e[0m  Force rerun: deleting existing results for $task_name"
            rm -rf "$output_dir/$task_name"
        else
            echo -e "\e[31m❌\e[0m  Skipping completed task $task_name"
            continue
        fi
    fi

    echo -e "\e[32m✅\e[0m  Submitting ${model_name}_${task_name}"

    cmd=$(generate "$task_name" "$lm_eval_model" "$lm_eval_args")
    echo "$cmd"
    eval "$cmd"

done 