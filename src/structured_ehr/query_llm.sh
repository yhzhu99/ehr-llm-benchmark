#!/bin/bash

# Basic configurations
OUTPUT_LOGITS=true
OUTPUT_PROMPTS=false

# Parameter options
MODEL_OPTIONS=(
    "DeepSeek-V3"

    "o3-mini-high"
    "chatgpt-4o-latest"

    "DeepSeek-R1-7B"
    "Gemma-3-4B"
    "HuatuoGPT-o1-7B"
    "OpenBioLLM"
    "Qwen2.5-7B"
)
DATASET_TASK_OPTIONS=(
    "tjh:mortality"
    "tjh:los"
    "mimic-iv:mortality"
    "mimic-iv:readmission"
)
UNIT_RANGE_OPTIONS=(false true)

# Compute total runs for progress display
TOTAL_RUNS=$((${#DATASET_TASK_OPTIONS[@]} * ${#UNIT_RANGE_OPTIONS[@]} * ${#MODEL_OPTIONS[@]}))
CURRENT_RUN=0

echo "Starting evaluation with ${TOTAL_RUNS} different configurations..."

# Iterate over dataset and task combinations
for DATASET_TASK in "${DATASET_TASK_OPTIONS[@]}"; do
    # Dataset and task
    IFS=":" read -r DATASET TASK <<< "$DATASET_TASK"
    for MODEL in "${MODEL_OPTIONS[@]}"; do
        for USE_UNIT_RANGE in "${UNIT_RANGE_OPTIONS[@]}"; do
            # Add counter
            CURRENT_RUN=$((CURRENT_RUN + 1))

            # Construct command
            CMD="python -m src.structured_ehr.query_llm -d ${DATASET} -t ${TASK} -m ${MODEL}"

            # Add parameters
            if [ "$USE_UNIT_RANGE" = true ]; then
              CMD="${CMD} -u -r -n 1"
            fi

            # Add output options
            if [ "$OUTPUT_LOGITS" = true ]; then
              CMD="${CMD} --output_logits"
            fi

            if [ "$OUTPUT_PROMPTS" = true ]; then
              CMD="${CMD} --output_prompts"
            fi

            # Print the counter
            echo "[$CURRENT_RUN/$TOTAL_RUNS] Running configuration..."

            # Print the command
            echo "Command: $CMD"

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
done

echo "All evaluations completed!"