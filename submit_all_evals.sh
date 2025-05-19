#!/bin/bash

if [ $# -ne 2 ]; then
    echo "Usage: $0 <npu_num> <job_num>"
    exit 1
fi

vc submit -p pdgpu-aispeech-ai -i hub.szaic.com/hpc/ai_nlp_base-openrlhf_vllm-0.7.3rc2:cann8.1rc1-alpah001 \
    -c $(( $1 * 20 )) -m $(( $1 * 120 ))G -g $1 \
    --job "eval_qw2.5moe" JOB=1:"$2" ../logs/submit_all_evals.JOB.log \
    --cmd "
        source /usr/local/Ascend/ascend-toolkit/set_env.sh && 
        source /usr/local/Ascend/nnal/atb/set_env.sh && 
        bash all_evals.sh && 
        echo \"All evaluations submitted to $1 NPUs\"
    "
