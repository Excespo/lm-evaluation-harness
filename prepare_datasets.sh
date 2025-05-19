#!/bin/bash

# 脚本目标：
# 1. 根据预定义的hf2local映射，使用HFD工具下载指定的数据集到新的本地目录。
# 2. 校验数据集是否成功下载（检查目标目录是否存在且非空）。
# 3. 更新 lm_eval/tasks 目录下的所有YAML配置文件，将其中的旧数据集路径替换为新的数据集路径。
#
# 使用此脚本可以帮助快速在新的工作环境中准备lm-evaluation-harness所需的数据集和配置。

# --- 参数校验 ---
if [ "$#" -ne 2 ]; then
    echo "错误: 参数数量不正确。"
    echo "用法: $0 <original_local_dataset_dir> <new_local_dataset_dir>"
    echo "示例: $0 /old/data/path /new/data/path"
    exit 1
fi

ORIGINAL_LOCAL_DATASET_DIR="$1"
NEW_LOCAL_DATASET_DIR="$2"
# 假设lm-evaluation-harness的tasks目录在当前工作区的lm_eval/tasks下
# 如果您的lm-evaluation-harness项目结构不同，请修改此路径
LM_EVAL_TASKS_DIR="lm_eval/tasks"

# HFD工具路径 (请根据您的实际安装路径修改)
HFD_TOOL_PATH="/aistor/aispeech/hpc_stor01/home/mada00sx/tools/hfd.sh"

if [ ! -f "$HFD_TOOL_PATH" ]; then
    echo "错误: HFD工具未在指定路径找到: $HFD_TOOL_PATH"
    echo "请检查HFD_TOOL_PATH变量或安装HFD。"
    exit 1
fi

# --- HuggingFace名称到本地目录名的映射 ---
# 请根据 `ls ~/data/yijie.luo/lm_eval` 的输出和实际的HuggingFace数据集ID来填充和调整此列表。
# key: HuggingFace Hub上的数据集名称 (例如 "openai/humaneval", "cais/mmlu")
# value: 您希望在本地使用的目录名 (例如 "humaneval", "mmlu_no_train")
declare -A hf2local=(
    ["openai/humaneval"]="humaneval"
    ["google/mbpp"]="mbpp"
    ["gsm8k"]="gsm8k" # 主版本通常是 'main'
    # ["allenai/ai2_arc"]="arc" # ARC (Challenge) - 注意ARC有多个子集，如ARC-Easy, ARC-Challenge
    ["allenai/arc"]="arc" # ARC (Challenge)
    ["winogrande"]="winogrande" # 通常是 "winogrande" 或 "facebook/winogrande"
    ["truthful_qa"]="truthful_qa" # 通常是 "truthful_qa"
    ["hellaswag"]="hellaswag"
    ["meta-math/MetaMathQA"]="MetaMathQA" # 示例，请确认实际HF ID
    # 根据 `ls ~/data/yijie.luo/lm_eval` 的输出添加更多映射
    # 例如:
    # ["cais/mmlu"]="mmlu_no_train" # 假设 mmlu_no_train 是您本地的MMLU目录名
    # ["hendrycks/competition_math"]="hendrycks_math"
    # ["agieval"]="agieval" # 需要确认agieval对应的HF数据集是什么
    # ...等等
)

echo "HuggingFace到本地名称的映射 (hf2local):"
for hf_name in "${!hf2local[@]}"; do
    echo "  '${hf_name}' -> '${hf2local[$hf_name]}'"
done
echo "请检查以上映射是否正确，并根据需要修改脚本中的 'hf2local' 数组。"
echo "按 Enter 继续，或按 Ctrl+C 中断..."
read -r

# --- 下载数据集 ---
echo ""
echo "开始下载数据集到 '$NEW_LOCAL_DATASET_DIR'..."
mkdir -p "$NEW_LOCAL_DATASET_DIR" #确保目标根目录存在

missing_datasets=0
for hf_name in "${!hf2local[@]}"; do
    local_name="${hf2local[$hf_name]}"
    target_dataset_dir="$NEW_LOCAL_DATASET_DIR/$local_name"

    echo ""
    echo "处理数据集: HuggingFace名='${hf_name}', 本地名='${local_name}'"
    echo "目标下载路径: '${target_dataset_dir}'"

    if [ -d "${target_dataset_dir}" ] && [ "$(ls -A "${target_dataset_dir}")" ]; then
        echo "  目录 '${target_dataset_dir}' 已存在且非空，跳过下载。"
        continue
    fi

    mkdir -p "${target_dataset_dir}"
    echo "  执行下载: $HFD_TOOL_PATH --datasets \"${hf_name}\" --local_dir \"${target_dataset_dir}\" --tool aria2c -x 10"
    # shellcheck disable=SC2086
    if "$HFD_TOOL_PATH" --datasets "${hf_name}" --local_dir "${target_dataset_dir}" --tool aria2c -x 10; then
        echo "  成功下载 '${hf_name}' 到 '${target_dataset_dir}'."
        if [ ! -d "${target_dataset_dir}" ] || [ -z "$(ls -A "${target_dataset_dir}")" ]; then
             echo "  警告: '${hf_name}' 下载后目录 '${target_dataset_dir}' 不存在或为空。"
             echo "  请检查HFD工具的输出和目标目录。"
             missing_datasets=$((missing_datasets + 1))
        fi
    else
        echo "  错误: 下载 '${hf_name}' 失败。"
        missing_datasets=$((missing_datasets + 1))
    fi
done

if [ "$missing_datasets" -gt 0 ]; then
    echo ""
    echo "警告: 有 ${missing_datasets} 个数据集未能成功下载或验证。请检查上面的日志。"
    echo "按 Enter 继续进行YAML文件路径替换，或按 Ctrl+C 中断..."
    read -r
else
    echo ""
    echo "所有已配置的数据集均已处理完毕。"
fi


# --- 验证是否有缺失的数据集 (根据hf2local映射) ---
# 此部分已集成到下载循环中


# --- 修改 lm_eval/tasks YAML 文件中的 dataset_path ---
echo ""
echo "开始更新 YAML 文件中的 dataset_path..."
echo "查找 '${LM_EVAL_TASKS_DIR}' 目录下的 YAML 文件..."
echo "将 '${ORIGINAL_LOCAL_DATASET_DIR}' 替换为 '${NEW_LOCAL_DATASET_DIR}'"

if [ ! -d "$LM_EVAL_TASKS_DIR" ]; then
    echo "错误: Tasks 目录 '$LM_EVAL_TASKS_DIR' 未找到。请检查 LM_EVAL_TASKS_DIR 变量。"
    exit 1
fi

# 使用find查找所有yaml和yml文件，然后用sed替换
# 注意: sed的替换分隔符使用'#'以避免路径中的'/'导致问题
yaml_files_found=0
# shellcheck disable=SC2038
for yaml_file in $(find "$LM_EVAL_TASKS_DIR" -type f \( -name "*.yaml" -o -name "*.yml" \)); do
    echo "  处理文件: $yaml_file"
    # 使用 grep 检查文件是否包含旧路径，如果包含则执行替换
    if grep -qF "$ORIGINAL_LOCAL_DATASET_DIR" "$yaml_file"; then
        # 创建备份
        cp "$yaml_file" "${yaml_file}.bak"
        # 执行替换
        # sed -i.bak "s#${ORIGINAL_LOCAL_DATASET_DIR}#${NEW_LOCAL_DATASET_DIR}#g" "$yaml_file" # macOS sed
        sed -i "s#${ORIGINAL_LOCAL_DATASET_DIR}#${NEW_LOCAL_DATASET_DIR}#g" "$yaml_file" # GNU sed
        if [ $? -eq 0 ]; then
            echo "    成功替换路径。备份文件为 ${yaml_file}.bak"
        else
            echo "    错误: 替换路径失败。请检查文件权限或sed命令。"
            # 如果替换失败，可以考虑恢复备份
            # mv "${yaml_file}.bak" "$yaml_file"
        fi
    else
        echo "    未找到旧路径 '${ORIGINAL_LOCAL_DATASET_DIR}'，跳过替换。"
    fi
    yaml_files_found=$((yaml_files_found + 1))
done

if [ "$yaml_files_found" -eq 0 ]; then
    echo "未在 '${LM_EVAL_TASKS_DIR}' 目录下找到任何 YAML 文件。"
else
    echo "YAML 文件处理完毕。共处理 ${yaml_files_found} 个文件（可能包含未执行替换的文件）。"
fi

echo ""
echo "脚本执行完毕。"
echo "请检查 '$NEW_LOCAL_DATASET_DIR' 中的数据集和 '${LM_EVAL_TASKS_DIR}' 中的YAML文件。"
echo "重要提示: 您可能需要根据 'ls ~/data/yijie.luo/lm_eval' 的输出和实际的HuggingFace数据集ID来仔细配置脚本中的 'hf2local' 数组，以确保所有需要的数据集都得到处理。"

