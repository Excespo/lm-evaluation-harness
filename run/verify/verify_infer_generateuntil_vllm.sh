#!/bin/bash
set -e
source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /usr/local/Ascend/nnal/atb/set_env.sh
export HF_ENDPOINT=https://hf-mirror.com 
export PYTHONPATH=$PYTHONPATH:$(pwd)
export PATH=$PATH:~/.local/bin
export HCCL_CONNECT_TIMEOUT=4800
export HCCL_EXEC_TIMEOUT=4800
export HF_ALLOW_CODE_EVAL=1
export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
export ASCEND_RT_VISIBLE_DEVICES=3,4
dp_size=$(python -c 'import os; devices=os.environ["ASCEND_RT_VISIBLE_DEVICES"]; print(len(devices.split(",")))')
echo "Detected ASCEND_RT_VISIBLE_DEVICES=$ASCEND_RT_VISIBLE_DEVICES, dp size then is $dp_size"
echo "🎯 Using vllm monkey patch from file path_to_modeling_monkey_patch=/aistor/sjtu/hpc_stor01/home/luoyijie/src/lm-evaluation-harness/fpm_vllm_modeling_patch.py,path_to_config_monkey_patch=/aistor/sjtu/hpc_stor01/home/luoyijie/src/lm-evaluation-harness/fpm_vllm_config_patch.py'"


# lm_eval --model vllm\
#     --model_args "pretrained=/aistor/sjtu/hpc_stor01/home/luoyijie/src/FPM/outputs/ckpts/Qwen2.5-1.5B-Cpt-culture_final,dtype=bfloat16,max_model_len=4096,tensor_parallel_size=1,gpu_memory_utilization=0.95,data_parallel_size=$dp_size" \
#     --tasks nq_open --num_fewshot 0 \
#     --log_samples --output_path outputs/verify_infer_vllm/nq_open
echo "7.65% is the exact match acc of culture cpt model"

lm_eval --model vllm \
    --model_args "pretrained=/aistor/sjtu/hpc_stor01/home/luoyijie/src/FPM/outputs/ckpts/init/Qwen2.5-MoE-6.8B-en-wiki-init-0+1-clone-culture-weights,dtype=bfloat16,max_model_len=4096,tensor_parallel_size=1,gpu_memory_utilization=0.95,data_parallel_size=$dp_size,path_to_modeling_monkey_patch=/aistor/sjtu/hpc_stor01/home/luoyijie/src/lm-evaluation-harness/fpm_vllm_modeling_patch.py,path_to_config_monkey_patch=/aistor/sjtu/hpc_stor01/home/luoyijie/src/lm-evaluation-harness/fpm_vllm_config_patch.py,use_experts=culture" \
    --tasks nq_open --num_fewshot 0 \
    --log_samples --output_path outputs/verify_infer_vllm/nq_open 
    # --use_experts