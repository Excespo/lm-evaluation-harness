#!/bin/bash

model=/aistor/sjtu/hpc_stor01/home/luoyijie/ckpts/fpm/Qwen2.5-MoE-5.7B-wiki-init-clone-mean-common-weights
step=
domains=(
    "geography"
    "history"
    "politics"
    "literature"
)
mkdir -p logs

echo "Starting sequential evaluation of all tasks..."
echo "----------------------------------------------------"

# --- Run 1: Baseline Task (fpm_wiki) ---
task_name_baseline="fpm_wiki"
log_file_baseline="logs/run_${model##*/}_${task_name_baseline}.log"

echo "Running task: ${task_name_baseline} (baseline)"
echo "  -> Log file: ${log_file_baseline}"
# bash run/run_moe.sh "$model" "$step" "$task_name_baseline" --force-rerun > "$log_file_baseline" 2>&1
echo "Task finished."
echo "----------------------------------------------------"


# --- Run 2: Domain-Specific Tasks ---
for domain in "${domains[@]}"; do
    task_name_domain="fpm_wiki_$domain"
    log_file_domain="logs/run_${model##*/}_${task_name_domain}_use_${domain}.log"
    
    echo "Running task: ${task_name_domain} (expert: ${domain})"
    echo "  -> Log file: ${log_file_domain}"
    bash run/run_moe.sh --use_experts "$domain" "$model" "$step" "$task_name_domain" > "$log_file_domain" 2>&1
    echo "Task finished."
    echo "----------------------------------------------------"
done

echo
echo -e "\e[1;34mAll tasks have completed sequentially.\e[0m"
echo "Check the 'logs/' directory for detailed output from each run."
