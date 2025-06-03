#!/bin/bash

source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /usr/local/Ascend/nnal/atb/set_env.sh
export HF_ENDPOINT=https://hf-mirror.com
export PYTHONPATH=$PYTHONPATH:$(pwd)
export PATH=$PATH:~/.local/bin
export HCCL_CONNECT_TIMEOUT=4800
export HCCL_EXEC_TIMEOUT=4800

model_args=(
    "pretrained=/aistor/sjtu/hpc_stor01/home/luoyijie/ckpts/fpm/Qwen2.5-MoE-5.7B-clone-math-attn-init"
    "dtype=bfloat16"
    "max_model_len=4096"
    "tensor_parallel_size=1"
    "gpu_memory_utilization=0.7"
    "path_to_modeling_monkey_patch=/aistor/sjtu/hpc_stor01/home/luoyijie/src/FPM/src/model/vllm_moe_monkey_patch.py"
    "path_to_config_monkey_patch=/aistor/sjtu/hpc_stor01/home/luoyijie/src/FPM/src/model/config_patch.py"
)
    
lm_eval --model vllm \
    --model_args "$(IFS=,; echo "${model_args[*]}")" \
    --tasks gsm8k --num_fewshot 8 --batch_size 1 \
    --gen_kwargs max_new_tokens=512 \
    --log_samples --output_path use_math_vllm