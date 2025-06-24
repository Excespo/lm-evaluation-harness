#!/bin/bash

export PATH=$PATH:~/.local/bin
export HCCL_CONNECT_TIMEOUT=3000
export HCCL_EXEC_TIMEOUT=3000
export HF_ALLOW_CODE_EVAL=1
export HF_DATASET_CACHE=/aistor/sjtu/hpc_stor01/home/luoyijie/data/lm_eval/cache
export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1

accelerate launch -m lm_eval --model hf --model_args pretrained=/aistor/sjtu/hpc_stor01/home/luoyijie/src/LongContext/outputs/ckpts/Qwen2.5-1.5B-Cpt-Wiki-NQ_History-10epochs-32BS/final,dtype=auto --tasks nq_open_history --include_path /aistor/sjtu/hpc_stor01/home/luoyijie/src/lm-evaluation-harness/fpm_tasks --log_samples --show_config --output_path outputs/Qwen2.5-1.5B-Cpt-Wiki-NQ_History-10epochs-32BS_final/hf/nq_open_history --num_fewshot 0 --batch_size 32

accelerate launch -m lm_eval --model hf --model_args pretrained=/aistor/sjtu/hpc_stor01/home/luoyijie/src/LongContext/outputs/ckpts/Qwen2.5-1.5B-Cpt-Wiki-NQ_Geography-10epochs-32BS/final,dtype=auto --tasks nq_open_geography --include_path /aistor/sjtu/hpc_stor01/home/luoyijie/src/lm-evaluation-harness/fpm_tasks --log_samples --show_config --output_path outputs/Qwen2.5-1.5B-Cpt-Wiki-NQ_Geography-10epochs-32BS_final/hf/nq_open_geography --num_fewshot 0 --batch_size 32

accelerate launch -m lm_eval --model hf --model_args pretrained=/aistor/sjtu/hpc_stor01/home/luoyijie/src/LongContext/outputs/ckpts/Qwen2.5-1.5B-Cpt-Wiki-NQ_Literature-10epochs-32BS/final,dtype=auto --tasks nq_open_literature --include_path /aistor/sjtu/hpc_stor01/home/luoyijie/src/lm-evaluation-harness/fpm_tasks --log_samples --show_config --output_path outputs/Qwen2.5-1.5B-Cpt-Wiki-NQ_Literature-10epochs-32BS_final/hf/nq_open_literature --num_fewshot 0 --batch_size 32

accelerate launch -m lm_eval --model hf --model_args pretrained=/aistor/sjtu/hpc_stor01/home/luoyijie/src/LongContext/outputs/ckpts/Qwen2.5-1.5B-Cpt-Wiki-NQ_Politics-10epochs-32BS/final,dtype=auto --tasks nq_open_politics --include_path /aistor/sjtu/hpc_stor01/home/luoyijie/src/lm-evaluation-harness/fpm_tasks --log_samples --show_config --output_path outputs/Qwen2.5-1.5B-Cpt-Wiki-NQ_Politics-10epochs-32BS_final/hf/nq_open_politics --num_fewshot 0 --batch_size 32