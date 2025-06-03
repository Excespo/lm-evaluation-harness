#!/bin/bash

source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /usr/local/Ascend/nnal/atb/set_env.sh
export HF_ENDPOINT=https://hf-mirror.com 
export PYTHONPATH=$PYTHONPATH:$(pwd)
export PATH=$PATH:~/.local/bin
export HCCL_CONNECT_TIMEOUT=4800
export HCCL_EXEC_TIMEOUT=4800
modeling_patch=/aistor/sjtu/hpc_stor01/home/luoyijie/src/FPM/src/model/moe_monkey_patch.py
config_patch=/aistor/sjtu/hpc_stor01/home/luoyijie/src/FPM/src/model/config_patch.py

accelerate launch -m lm_eval --model hf --batch_size 8 \
    --model_args pretrained=/aistor/sjtu/hpc_stor01/home/luoyijie/ckpts/fpm/Qwen2.5-MoE-5.7B-clone_unbalanced-mix-with-tag-40B-bf16_step-12000,dtype=auto,path_to_modeling_monkey_patch=$modeling_patch,path_to_config_monkey_patch=$config_patch \
    --tasks fpm_medical  --num_fewshot 5 \
    --log_samples --output_path use_medical --use_experts medical