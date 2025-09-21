#!/bin/bash

# 基本配置
LOG_DIR="logs/running_logs/structured_ehr/dl_models"
MAX_JOBS=12 # 设置最大并发任务数

# 参数选项
MODEL_OPTIONS=(
    "CatBoost"
    "DT"
    "RF"
    "XGBoost"
    "GRU"
    "LSTM"
    "Transformer"
    "RNN"
    "AdaCare"
    "AICare"
    "ConCare"
    "GRASP"
)
DATASET_TASK_OPTIONS=(
    "tjh:mortality"
    "tjh:los"
    "mimic-iv:mortality"
    "mimic-iv:readmission"
)
SHOT_OPTIONS=(
    "few"
    "full"
)

# --- 主脚本从这里开始 ---

# 如果日志目录不存在，则创建
mkdir -p "$LOG_DIR"
echo "日志文件将保存在 ${LOG_DIR}/"

# 用于存储命令及其对应日志文件的数组
COMMANDS=()
LOG_FILES=()

# 生成所有命令并存入数组
echo "正在生成所有命令..."
for DATASET_TASK in "${DATASET_TASK_OPTIONS[@]}"; do
    # 分割数据集和任务
    IFS=":" read -r DATASET TASK <<< "$DATASET_TASK"
    for SHOT in "${SHOT_OPTIONS[@]}"; do
        for MODEL in "${MODEL_OPTIONS[@]}"; do
            # 为当前数据集、任务和shot类型组合所有模型
            # 注意：这里将所有模型作为一个参数列表传递给脚本
            CMD="python -m src.structured_ehr.train_dl -d ${DATASET} -t ${TASK} -s ${SHOT} -m ${MODEL}"

            # 将完整构造的命令添加到数组
            COMMANDS+=("$CMD")

            # 生成对应的日志文件路径
            LOG_FILE="${LOG_DIR}/${DATASET}-${TASK}-${SHOT}-${MODEL}.log"
            LOG_FILES+=("$LOG_FILE")
        done
    done
done

# 获取命令总数
TOTAL_RUNS=${#COMMANDS[@]}
echo "准备开始评估，共计 ${TOTAL_RUNS} 个不同的配置..."

# 限制并发任务数，执行所有命令
for i in "${!COMMANDS[@]}"; do
    CMD="${COMMANDS[$i]}"
    LOG_FILE="${LOG_FILES[$i]}"
    CURRENT_RUN=$((i + 1))

    # 如果正在运行的任务数达到上限，则等待
    while (( $(jobs -p | wc -l) >= MAX_JOBS )); do
        wait -n # 等待任意一个后台任务完成
    done

    # 打印计数器和正在后台运行的命令
    echo "[$CURRENT_RUN/$TOTAL_RUNS] 正在后台运行。日志 -> ${LOG_FILE}"
    echo "  => ${CMD}"

    # 在后台执行命令，并将所有输出重定向到日志文件
    (
        # 实际执行并重定向输出
        eval "$CMD" > "$LOG_FILE" 2>&1

        # 以下部分在命令结束后执行
        # 其输出将显示在主控制台，而不是日志文件中
        if [ $? -eq 0 ]; then
          echo "[$CURRENT_RUN/$TOTAL_RUNS] 成功: 任务已完成。日志: ${LOG_FILE}"
        else
          echo "[$CURRENT_RUN/$TOTAL_RUNS] 失败: 任务出错。日志: ${LOG_FILE}"
        fi
        echo "----------------------------------------"
    ) &

done

# 等待所有剩余的后台任务完成
echo "所有命令已分发。正在等待剩余的后台任务结束..."
wait

echo "所有训练已完成！"