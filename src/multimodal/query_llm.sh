#!/bin/bash

# Basic configurations
OUTPUT_LOGITS=true
OUTPUT_PROMPTS=true
LOG_DIR="logs/running_logs/multimodal"

# Parameter options
MODEL_OPTIONS=(
    "deepseek-v3-chat"
    "deepseek-v3-reasoner"
    "deepseek-r1"

    "chatgpt-4o-latest"
    "o3-mini-high"
    "gpt-5-chat-latest"

    "DeepSeek-R1-7B"
    "Gemma-3-4B"
    "HuatuoGPT-o1-7B"
    "OpenBioLLM"
    "Qwen2.5-7B"
)
DATASET_TASK_OPTIONS=(
    "mimic-iv:mortality"
    "mimic-iv:readmission"
)
EHR_FORMAT_OPTIONS=(
  "list"
  "text"
)
NOTE_FIRST_OPTIONS=(
  "true"
  "false"
)

# --- Main script starts here ---

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
  for EHR_FORMAT in "${EHR_FORMAT_OPTIONS[@]}"; do
    for NOTE_FIRST in "${NOTE_FIRST_OPTIONS[@]}"; do
      for MODEL in "${MODEL_OPTIONS[@]}"; do
        # Construct command
        CMD="python -m src.multimodal.query_llm -d ${DATASET} -t ${TASK} -m ${MODEL}"

        # Add parameters based on USE_UNIT_RANGE
        CMD="${CMD} -u -r"
        UNIT_RANGE_SUFFIX="-unit-range"

        # Add EHR format
        CMD="${CMD} -f ${EHR_FORMAT}"
        EHR_FORMAT_SUFFIX="-${EHR_FORMAT}"

        # Add note first
        if [ "$NOTE_FIRST" = "true" ]; then
          CMD="${CMD} --note_first"
          NOTE_FIRST_SUFFIX="-note_first"
        else
          NOTE_FIRST_SUFFIX="-ehr_first"
        fi

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
        LOG_FILE="${LOG_DIR}/${DATASET}-${TASK}-${MODEL}${UNIT_RANGE_SUFFIX}${EHR_FORMAT_SUFFIX}${NOTE_FIRST_SUFFIX}.log"
        LOG_FILES+=("$LOG_FILE")
      done
    done
  done
done

# Get the total number of commands
TOTAL_RUNS=${#COMMANDS[@]}
MAX_JOBS=10 # Set the maximum number of concurrent jobs

echo "Starting evaluation with ${TOTAL_RUNS} different configurations..."

# Execute all commands with a limit on concurrent jobs
for i in "${!COMMANDS[@]}"; do
    CMD="${COMMANDS[$i]}"
    LOG_FILE="${LOG_FILES[$i]}"
    CURRENT_RUN=$((i + 1))

    # Wait if the number of running jobs reaches the maximum
    while (( $(jobs -p | wc -l) >= MAX_JOBS )); do
        wait -n # Wait for any background job to complete
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
