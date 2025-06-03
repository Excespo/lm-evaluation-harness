#!/bin/bash
if [ $# -lt 2 ]; then
    echo "Usage: $0 <ckpt_base_dir> <ckpt_name>"
    exit 1
fi
base=$1; ckpt=$2;

# interactive scripts have to do it first
# source /usr/local/Ascend/ascend-toolkit/set_env.sh
# source /usr/local/Ascend/nnal/atb/set_env.sh
export HCCL_CONNECT_TIMEOUT=3000
export HCCL_EXEC_TIMEOUT=3000
export HF_ALLOW_CODE_EVAL=1
export HF_DATASET_CACHE=/aistor/sjtu/hpc_stor01/home/luoyijie/data/lm_eval/cache
export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1

declare -A tasks=(
    ["nq_open"]="hf#--num_fewshot 3"
)

# build env for model
model_name=${1##*/}
if [ -z $2 ]; then
    # no intermediate ckpts, or subdirs, use directly base dir as ckpt dir
    ckpt=$1
    model_name=${model_name}
else
    ckpt=$1/$2
    model_name=${model_name}_$2
fi
output_dir=outputs/$model_name
mkdir -p $output_dir

# generate eval cmd
# task_name="$1"; lm_eval_model="$2"; lm_eval_args="$3";"
generate() { 

    if [ "$2" == "vllm" ]; then
        echo "Not supporting vllm"; exit
        launch="lm_eval --model vllm"
        model_args="pretrained=$ckpt,dtype=auto,max_model_len=4096,tensor_parallel_size=1,gpu_memory_utilization=0.7"
    elif [ "$2" == "hf" ]; then
        launch="accelerate launch -m lm_eval --model hf --batch_size 8"
        model_args="pretrained=$ckpt,dtype=auto"
    else
        return 1
    fi

    echo "$launch" \
        --model_args "$model_args" \
        --tasks "$1" \
        --log_samples \
        --show_config \
        --output_path "$output_dir/$1" \
        "$3"
}

# prepare-and-submit loop
for task_name in "${!tasks[@]}"; do

    IFS="#" read -r lm_eval_model lm_eval_args <<< "${tasks[$task_name]}"
    
    if compgen -G "$output_dir/$task_name/**/results_*.json" > /dev/null; then
        echo -e "\e[31m❌\e[0m Skipping completed task $task_name"
        continue
    fi

    echo -e "\e[32m✅\e[0m Submitting ${model_name}_${task_name}"

    cmd=$(generate "$task_name" "$lm_eval_model" "$lm_eval_args")
    echo "$cmd"
    eval "$cmd"

done