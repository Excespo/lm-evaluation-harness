#!/bin/bash
set -e
DOMAINS=("geography" "history" "politics" "literature")
CKPT_DIR=/aistor/sjtu/hpc_stor01/home/luoyijie/ckpts/fpm/Qwen2.5-MoE-5.7B-wiki-mean-common-32BS-func_tagged-router_loss
STEP=final_st

for d in "${DOMAINS[@]}"; do
    bash run/run.sh --force-rerun --task moe_stat --model hf $CKPT_DIR $STEP stats_pretrain_nq_open_$d
done
