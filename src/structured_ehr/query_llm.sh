#!/bin/bash

# Basic configurations
MODEL="DeepSeek"
N_SHOT=1
OUTPUT_LOGITS=true
OUTPUT_PROMPTS=true

# Parameter options
DATASET_TASK_OPTIONS=(
    "tjh:mortality"
    "mimic-iv:mortality"
    "mimic-iv:readmission"
)
UNIT_RANGE_OPTIONS=(false true)

# Compute total runs for progress display
TOTAL_RUNS=$((${#DATASET_TASK_OPTIONS[@]} * ${#UNIT_RANGE_OPTIONS[@]}))
CURRENT_RUN=0

echo "Starting evaluation with ${TOTAL_RUNS} different configurations..."

eval "cd src/structured_ehr"

# Iterate over dataset and task combinations
for DATASET_TASK in "${DATASET_TASK_OPTIONS[@]}"; do
    # Dataset and task
    IFS=":" read -r DATASET TASK <<< "$DATASET_TASK"
    for USE_UNIT_RANGE in "${UNIT_RANGE_OPTIONS[@]}"; do
        # Add counter
        CURRENT_RUN=$((CURRENT_RUN + 1))

        # Construct command
        CMD="python query_llm.py -d ${DATASET} -t ${TASK} -m ${MODEL}"

        # Add parameters
        if [ "$USE_UNIT_RANGE" = true ]; then
          CMD="${CMD} -u -r"
        fi

        # Add nshot options
        CMD="${CMD} --n_shot ${N_SHOT}"

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