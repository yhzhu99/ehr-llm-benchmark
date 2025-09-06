#!/bin/bash

# Basic configurations
OUTPUT_LOGITS=true
OUTPUT_PROMPTS=true
LOG_DIR="logs/running_logs"

# Parameter options
MODEL_OPTIONS=(
    "DeepSeek-V3"
    "DeepSeek-R1"

    "o3-mini-high"
    "chatgpt-4o-latest"
    "gpt-5-chat-latest"

    "DeepSeek-R1-7B"
    "Gemma-3-4B"
    "HuatuoGPT-o1-7B"
    "OpenBioLLM"
    "Qwen2.5-7B"
)
DATASET_TASK_OPTIONS=(
    "mimic-iii:mortality"
    "mimic-iv:mortality"
    "mimic-iv:readmission"
)

# Create log directory if it doesn't exist
mkdir -p "$LOG_DIR"
echo "Log files will be saved in ${LOG_DIR}/"

# Arrays to store commands and their corresponding log files
COMMANDS=()
LOG_FILES=()

# Generate all commands and store them in the arrays
echo "Generating all commands..."
for DATASET_TASK in "${DATASET_TASK_OPTIONS[@]}"; do
    # Dataset and task
    IFS=":" read -r DATASET TASK <<< "$DATASET_TASK"
    for MODEL in "${MODEL_OPTIONS[@]}"; do
        # Construct command
        CMD="python -m src.unstructured_note.llm_generation_setting.query_llm -d ${DATASET} -t ${TASK} -m ${MODEL}"

        # Add output options
        if [ "$OUTPUT_LOGITS" = true ]; then
          CMD="${CMD} --output_logits"
        fi

        if [ "$OUTPUT_PROMPTS" = true ]; then
          CMD="${CMD} --output_prompts"
        fi

        # Add the fully constructed command to the array
        COMMANDS+=("$CMD")

        # Generate a corresponding log file path
        LOG_FILE="${LOG_DIR}/${DATASET}-${TASK}-${MODEL}.log"
        LOG_FILES+=("$LOG_FILE")
    done
done

# Get the total number of commands
TOTAL_RUNS=${#COMMANDS[@]}
MAX_JOBS=4 # Set the maximum number of concurrent jobs

echo "Starting evaluation with ${TOTAL_RUNS} different configurations..."

# Execute all commands with a limit on concurrent jobs
for i in "${!COMMANDS[@]}"; do
    CMD="${COMMANDS[$i]}"
    LOG_FILE="${LOG_FILES[$i]}"
    CURRENT_RUN=$((i + 1))

    # Wait if the number of running jobs reaches the maximum
    while [[ $(jobs -p | wc -l) -ge $MAX_JOBS ]]; do
        wait -n
    done

    # Print the counter and the command being run
    echo "[$CURRENT_RUN/$TOTAL_RUNS] Running in background. Log -> ${LOG_FILE}"
    echo "  => ${CMD}"

    # Execute command in the background, redirecting all output to the log file
    (
        # The actual execution with redirection
        eval "$CMD" > "$LOG_FILE" 2>&1

        # This part will run after the command finishes.
        # Its output will go to the main console, not the log file.
        if [ $? -eq 0 ]; then
          echo "[$CURRENT_RUN/$TOTAL_RUNS] SUCCESS: Job finished. Log: ${LOG_FILE}"
        else
          echo "[$CURRENT_RUN/$TOTAL_RUNS] FAILED: Job finished with an error. Log: ${LOG_FILE}"
        fi
        echo "----------------------------------------"
    ) &

done

# Wait for all remaining background jobs to complete
echo "All commands have been dispatched. Waiting for remaining background jobs to finish..."
wait

echo "All evaluations completed!"