#!/bin/bash

# 设置要遍历的参数
MODEL_OPTIONS=(
    "GRU"
    "AdaCare"
)
DATASET_TASK_OPTIONS=(
    "tjh:outcome"
    "mimic-iv:outcome"
    "mimic-iv:readmission"
)

# 计算总运行次数用于显示进度
TOTAL_RUNS=$((${#MODEL_OPTIONS[@]} * ${#DATASET_TASK_OPTIONS[@]}))
CURRENT_RUN=0

echo "Starting evaluation with ${TOTAL_RUNS} different configurations..."

# 遍历所有组合
for DATASET_TASK in "${DATASET_TASK_OPTIONS[@]}"; do
    # 解析数据集和任务
    IFS=":" read -r DATASET TASK <<< "$DATASET_TASK"
    for MODEL in "${MODEL_OPTIONS[@]}"; do
        # 增加计数器
        CURRENT_RUN=$((CURRENT_RUN + 1))

        # 构建命令
        CMD="python train_dl.py -d ${DATASET} -t ${TASK} -m ${MODEL}"

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

echo "All training completed!"