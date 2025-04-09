#!/bin/bash

# Basic configurations
OUTPUT_LOGITS=true
OUTPUT_PROMPTS=true

# Parameter options
MODEL_OPTIONS=(
    "DeepSeek"
)
DATASET_TASK_OPTIONS=(
    "mimic-iv:mortality"
    "mimic-iv:readmission"
)

# Compute total runs for progress display
TOTAL_RUNS=$((${#DATASET_TASK_OPTIONS[@]} * ${#MODEL_OPTIONS[@]}))
CURRENT_RUN=0

echo "Starting evaluation with ${TOTAL_RUNS} different configurations..."

eval "cd src/unstructured_note/llm_generation_setting"

# Iterate over dataset and task combinations
for DATASET_TASK in "${DATASET_TASK_OPTIONS[@]}"; do
    # Dataset and task
    IFS=":" read -r DATASET TASK <<< "$DATASET_TASK"
    for MODEL in "${MODEL_OPTIONS[@]}"; do
        # Add counter
        CURRENT_RUN=$((CURRENT_RUN + 1))

        # Construct command
        CMD="python query_llm.py -d ${DATASET} -t ${TASK} -m ${MODEL}"

        # Add output options
        if [ "$OUTPUT_LOGITS" = true ]; then
          CMD="${CMD} --output_logits"
        fi

        if [ "$OUTPUT_PROMPTS" = true ]; then
          CMD="${CMD} --output_prompts"
        fi

        # Print the counter
        echo "[$CURRENT_RUN/$TOTAL_RUNS] Running configuration..."

        # Execute command
        eval "$CMD"

        # Check if the command was successful
        if [ $? -eq 0 ]; then
          echo "[$CURRENT_RUN/$TOTAL_RUNS] Successfully completed..."
        else
          echo "[$CURRENT_RUN/$TOTAL_RUNS] Failed..."
        fi

        echo "----------------------------------------"
    done
done

echo "All evaluations completed!"