#!/bin/bash

# 设置基本参数
MODEL="DeepSeek"
N_SHOT=0
OUTPUT_LOGITS=false
OUTPUT_PROMPTS=true

# 设置要遍历的参数
DATASET_TASK_OPTIONS=(
    "tjh:outcome"
    "mimic-iv:outcome"
    "mimic-iv:readmission"
)
UNIT_RANGE_OPTIONS=(false true)

# 计算总运行次数用于显示进度
TOTAL_RUNS=$((${#DATASET_TASK_OPTIONS[@]} * ${#UNIT_RANGE_OPTIONS[@]}))
CURRENT_RUN=0

echo "Starting evaluation with ${TOTAL_RUNS} different configurations..."

# 遍历所有组合
for DATASET_TASK in "${DATASET_TASK_OPTIONS[@]}"; do
    # 解析数据集和任务
    IFS=":" read -r DATASET TASK <<< "$DATASET_TASK"
    for USE_UNIT_RANGE in "${UNIT_RANGE_OPTIONS[@]}"; do
        # 增加计数器
        CURRENT_RUN=$((CURRENT_RUN + 1))

        # 构建命令
        CMD="python query_llm.py -d ${DATASET} -t ${TASK} -m ${MODEL}"

        # 添加条件参数
        if [ "$USE_UNIT_RANGE" = true ]; then
          CMD="${CMD} -u -r"
        fi

        # 添加输出配置
        if [ "$OUTPUT_LOGITS" = true ]; then
          CMD="${CMD} --output_logits"
        fi

        if [ "$OUTPUT_PROMPTS" = true ]; then
          CMD="${CMD} --output_prompts"
        fi

        # 打印进度和当前配置
        echo "[$CURRENT_RUN/$TOTAL_RUNS] Running configuration..."

        # 执行命令
        eval "$CMD"

        # 检查命令执行状态
        if [ $? -eq 0 ]; then
          echo "[$CURRENT_RUN/$TOTAL_RUNS] Successfully completed..."
        else
          echo "[$CURRENT_RUN/$TOTAL_RUNS] Failed..."
        fi

        echo "----------------------------------------"
    done
done

echo "All evaluations completed!"