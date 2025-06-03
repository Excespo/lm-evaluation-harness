#!/bin/bash
if [ $# -lt 3 ]; then
    echo "Usage: $0 <ckpt_base_dir> <ckpt_name> <npu>"
    exit 1
fi
base=$1; ckpt=$2; npu=$3

declare -A tasks=(
    # # GENERALIST
    # ["fpm_generalist"]="hf"#""
    # # ["hellaswag"]="vllm"#"--num_fewshot 10"
    # # ["agieval"]="vllm"#"--num_fewshot 3" 
    # # KNOWLEDGE
    # ["mmlu"]="hf"#"--num_fewshot 5"
    # ["mmlu_generative"]="hf"#"--num_fewshot 5"
    # ["mmlu_pro"]="vllm"#"--num_fewshot 5"
    # # CODE
    # ["humaneval"]="vllm"#"--gen_kwargs do_sample=True,temperature=0.1 --trust_remote_code --confirm_run_unsafe_code"
    # ["humaneval_plus"]="vllm"#"--gen_kwargs do_sample=True,temperature=0.1 --trust_remote_code --confirm_run_unsafe_code"
    # ["mbpp"]="vllm"#"--num_fewshot 0 --trust_remote_code --confirm_run_unsafe_code"
    # ["mbpp_plus"]="vllm"#"--num_fewshot 0 --trust_remote_code --confirm_run_unsafe_code"
    # ["fpm_code"]="hf"#"--confirm_run_unsafe_code"
    # ["fpm_mbpp"]="vllm"#"--confirm_run_unsafe_code"
    # # MATH
    # ["gsm8k_cot"]="vllm"#"--num_fewshot 8 --gen_kwargs max_gen_toks=512"
    # ["gsm8k"]="vllm"#"--num_fewshot 8 --gen_kwargs max_gen_toks=512"
    # ["gpqa_diamond_cot_n_shot"]="vllm"#"--num_fewshot 4"
    # ["hendrycks_math"]="vllm"#"--num_fewshot 4"
    # ["math_500"]="vllm"#""
    # ["fpm_gsm8k"]="vllm"#"--num_fewshot 4"
    # ["cmath"]="vllm"#""
    # ["fpm_hendrycks_math"]="vllm"#""
    # ["fpm_gaokao_mathqa"]="vllm"#""
    # ["fpm_gaokao_mathcloze"]="vllm"#""
    # ["fpm_math"]="hf"#""
    # # MEDICAL
    # ["fpm_medical"]="hf"#"--num_fewshot 5"
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
# task_name="$1"; lm_eval_model="$2"; lm_eval_args="$3"; npu="$4"
generate() { 

    if [ "$2" == "vllm" ]; then
        launch="lm_eval --model vllm"
        model_args="pretrained=$ckpt,dtype=auto,max_model_len=4096,tensor_parallel_size=1,gpu_memory_utilization=0.7"
    elif [ "$2" == "hf" ]; then
        launch="accelerate launch -m lm_eval --model hf --batch_size 8"
        model_args="pretrained=$ckpt,dtype=auto"
    else
        return 1
    fi
    
    setup_npu="source /usr/local/Ascend/ascend-toolkit/set_env.sh && source /usr/local/Ascend/nnal/atb/set_env.sh"
    setup_eval="export HCCL_CONNECT_TIMEOUT=3000 && export HF_ALLOW_CODE_EVAL=1 && export HF_DATASET_CACHE=/aistor/sjtu/hpc_stor01/home/luoyijie/data/lm_eval/cache && export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1"
    echo "$setup_npu && $setup_eval && $launch" \
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

    vc submit \
        --partition pdgpu-aispeech-ai \
        --image hub.szaic.com/hpc/ai_nlp_base-openrlhf_vllm-0.7.3:cann8.1rc1-alpha001 \
        --cpu-per-task $((16*$npu)) \
        --mem-per-task $((48*$npu))G \
        --gpu-per-task $npu \
        JOB=1:1 logs/${model_name}_${task_name}.JOB.log \
        --cmd "$(generate "$task_name" "$lm_eval_model" "$lm_eval_args" "$npu")"

done

# track=$logs_dir/track_${model_name}.log
# # (
# #     while true; do
# #         echo -e "Track process started with PID: $$, logging to $track, query track process with \`pgrep -af "prepare_and_submit_tasks.sh"\`\n\n" > $track
# #         tail -n 5 $logs >> $track 2>&1;
# #         sleep 1;
# #     done
# # ) & 
# # echo Track process started with PID: $!, logging to $track
# TRACK_CMD="watch -n1 -t 'tail -n3 $logs'"
# echo -e "Track command written to $track: \n$TRACK_CMD"
# echo "$TRACK_CMD" > $track 2>&1
