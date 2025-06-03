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
    --model_args pretrained=/aistor/sjtu/hpc_stor01/home/luoyijie/ckpts/fpm/Qwen2.5-MoE-5.7B-clone-no-shared-math-as-common-init,dtype=bfloat16,path_to_modeling_monkey_patch=$modeling_patch,path_to_config_monkey_patch=$config_patch \
    --tasks mmlu  --num_fewshot 5 \
    --log_samples --output_path outputs/verify_infer/mmlu --use_experts math

accelerate launch -m lm_eval --model hf --batch_size 8 \
    --model_args pretrained=/aistor/sjtu/hpc_stor01/home/luoyijie/ckpts/Qwen2.5-Cpt-OWM-Math-1.5B-final,dtype=bfloat16 \
    --tasks mmlu  --num_fewshot 5 \
    --log_samples --output_path outputs/verify_infer/mmlu