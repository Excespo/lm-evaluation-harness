#!/bin/bash

# accelerate launch -m lm_eval --model hf --batch_size 8 --model_args pretrained=/aistor/sjtu/hpc_stor01/home/luoyijie/ckpts/Qwen2.5-Cpt-Medical-1.5B-final,dtype=auto --tasks nq_open --log_samples --show_config --output_path outputs/Qwen2.5-Cpt-Medical-1.5B-final/nq_open --num_fewshot 3
source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /usr/local/Ascend/nnal/atb/set_env.sh

export HCCL_CONNECT_TIMEOUT=3000
export HCCL_EXEC_TIMEOUT=3000
export HF_ALLOW_CODE_EVAL=1
export HF_DATASETS_CACHE=/aistor/sjtu/hpc_stor01/home/luoyijie/data/lm_eval/cache
export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
export PATH=/aistor/sjtu/hpc_stor01/home/luoyijie/.local/bin:$PATH

ckpt=/aistor/sjtu/hpc_stor01/home/luoyijie/ckpts/Qwen2.5-Cpt-Dolma_30B-General-1.5B-final
lm_eval --model vllm \
    --model_args pretrained=$ckpt,dtype=bfloat16,max_model_len=4096,tensor_parallel_size=1,gpu_memory_utilization=0.7 \
    --tasks triviaqa --num_fewshot 5 \
    --log_samples --output_path outputs/${ckpt##*/}/triviaqa
