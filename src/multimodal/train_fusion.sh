#!/bin/bash

# Parameter options
MODEL_DATASET_TASK_OPTIONS=(
    "AdaCare:GatorTron:mimic-iv:mortality"
    "LSTM:HuatuoGPT-o1-7B:mimic-iv:readmission"
)
FUSION_MODES=(
    "add"
    "concat"
    "attention"
    "cross_attention"
)

# --- Main script starts here ---

# Create log directory if it doesn't exist
LOG_DIR="logs/running_logs/multimodal/fusion_training"
mkdir -p "$LOG_DIR"
echo "Log files will be saved in ${LOG_DIR}/"

# Arrays to store commands and their corresponding log files
COMMANDS=()
LOG_FILES=()

# Generate all commands and store them in the arrays
echo "Generating all commands..."
for FUSION_MODE in "${FUSION_MODES[@]}"; do
    for MODEL_DATASET_TASK in "${MODEL_DATASET_TASK_OPTIONS[@]}"; do
        # Model, Dataset and task
        IFS=":" read -r EHR_MODEL NOTE_MODEL DATASET TASK <<< "$MODEL_DATASET_TASK"
        # Construct command
        CMD="python -m src.multimodal.fusion_training --ehr_model ${EHR_MODEL} --note_model ${NOTE_MODEL} --dataset ${DATASET} --task ${TASK} --fusion_mode ${FUSION_MODE}"

        CMD="${CMD} --hidden_dim 1024 --num_heads 8 --batch_size 32 --learning_rate 1e-4 --epochs 50 --patience 10"

        # Add the fully constructed command to the array
        COMMANDS+=("$CMD")

        # Generate a corresponding log file path
        LOG_FILE="${LOG_DIR}/${DATASET}-${TASK}-${EHR_MODEL}-${NOTE_MODEL}-${FUSION_MODE}.log"
        LOG_FILES+=("$LOG_FILE")
    done
done

# Get the total number of commands
TOTAL_RUNS=${#COMMANDS[@]}
MAX_JOBS=2 # Set the maximum number of concurrent jobs

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
