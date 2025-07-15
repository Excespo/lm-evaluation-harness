#!/bin/bash
set -e


# bash run/run.sh --task dense $(realpath ~/ckpts/huggingface/Qwen2.5-1.5B) ""

# bash run/run.sh --task dense $(realpath ~/ckpts/huggingface/Qwen2.5-3B) ""

CKPT_BASE_DIR=/aistor/sjtu/hpc_stor01/home/luoyijie/ckpts/fpm/en-wiki
DOMAINS=("culture" "entertainment" "geography" "history" "humanities" "science")

for d in ${DOMAINS[@]}; do
    # pretrained="$CKPT_BASE_DIR"/Qwen2.5-1.5B-Cpt-in-all-mlp-only
    # pretrained="$CKPT_BASE_DIR"/Qwen2.5-1.5B-Cpt-"$d"
    # pretrained="$CKPT_BASE_DIR"/Qwen2.5-1.5B-Cpt-"$d"-mlp-only

    steps=$(ls "$pretrained")
    # steps=$(ls "$pretrained" | grep -v final)
    # steps=("step_2000" "final")
    
    for step in ${steps[@]}; do
        bash run/run.sh --task dense "$pretrained" "$step" "$d"
    done
done
