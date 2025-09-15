#!/bin/bash

# Basic configurations
LOG_DIR="logs/running_logs/generate_embeddings"

# Define arrays of models and tasks
BERT_MODELS=("BERT" "ClinicalBERT" "BioBERT" "GatorTron" "Clinical-Longformer")
LLM_MODELS=("BioGPT" "meditron" "OpenBioLLM" "BioMistral" "GPT-2" "Qwen2.5-7B" "gemma-3-4b-pt" "HuatuoGPT-o1-7B" "DeepSeek-R1-Distill-Qwen-7B")
ALL_MODELS=("${BERT_MODELS[@]}" "${LLM_MODELS[@]}")
DATASET_OPTIONS=("mimic-iv" "mimic-iii")

# Create log directory if it doesn't exist
mkdir -p "$LOG_DIR"
echo "Log files will be saved in ${LOG_DIR}/"

# Arrays to store commands and their corresponding log files
COMMANDS=()
LOG_FILES=()

# Generate embeddings for BERT models
echo "Generating commands for BERT model embeddings..."
for MODEL in "${ALL_MODELS[@]}"; do
    for DATASET in "${DATASET_OPTIONS[@]}"; do
        CMD="python -m src.unstructured_note.freeze_setting.get_embeddings --model ${MODEL} --dataset ${DATASET}"
        COMMANDS+=("$CMD")
        # Replace '/' in model names for valid file names
        MODEL_SAFE_NAME=$(echo "$MODEL" | sed 's/\//-/g')
        LOG_FILE="${LOG_DIR}/embed-${MODEL_SAFE_NAME}-${DATASET}.log"
        LOG_FILES+=("$LOG_FILE")
    done
done

# Get the total number of commands
TOTAL_RUNS=${#COMMANDS[@]}
MAX_JOBS=10 # Set the maximum number of concurrent jobs (you can adjust this)

echo "Starting embedding generation with ${TOTAL_RUNS} different configurations..."

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
        eval "$CMD" > "$LOG_FILE" 2>&1

        if [ $? -eq 0 ]; then
          echo "[$CURRENT_RUN/$TOTAL_RUNS] SUCCESS: Job finished. Log: ${LOG_FILE}"
        else
          echo "[$CURRENT_RUN/$TOTAL_RUNS] FAILED: Job finished with an error. Log: ${LOG_FILE}"
        fi
        echo "----------------------------------------"
    ) &

done

# Wait for all remaining background jobs to complete
echo "All embedding generation commands have been dispatched. Waiting for remaining background jobs to finish..."
wait

echo "All embeddings generated successfully!"