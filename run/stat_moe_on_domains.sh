#!/bin/bash

# This script is for orchestrating statistics of MoE experts usage.
# It iterates through different tasks and expert configurations,
# calling the run_moe_stat.sh script for sequential execution with output redirection.

model=/aistor/sjtu/hpc_stor01/home/luoyijie/ckpts/fpm/Qwen2.5-MoE-5.7B-wiki-geography-common-32BS-func_tagged-router_loss
step=final
domains=(
    "geography"
    "history"
    "politics"
    "literature"
)
log_dir="logs"
mkdir -p "$log_dir"


# Test on the general fpm_wiki task without any expert mask
task="fpm_wiki"
log_file="${log_dir}/stat_${task}.log"
echo "Running statistics job for general task: $task. Logging to $log_file"
bash run/run_moe_stat.sh "$model" "$step" "$task" > "$log_file" 2>&1

# Test on domain-specific tasks, both with and without corresponding expert masks
for domain in "${domains[@]}"; do
    task_name="fpm_wiki_$domain"
    
    # Run statistics without specifying any experts
    log_file="${log_dir}/stat_${task_name}_no_expert.log"
    echo "Running stat job for $task_name without specified experts. Logging to $log_file"
    bash run/run_moe_stat.sh "$model" "$step" "$task_name" > "$log_file" 2>&1
    
    # Run statistics while specifying the corresponding domain expert
    log_file="${log_dir}/stat_${task_name}_with_expert_${domain}.log"
    echo "Running stat job for $task_name with specified expert: $domain. Logging to $log_file"
    bash run/run_moe_stat.sh "$model" "$step" "$task_name" "--use_experts $domain" > "$log_file" 2>&1
done

echo "All statistics jobs completed sequentially. Check the '$log_dir' directory for logs." 