#!/bin/bash

# Parameter options
MODEL_OPTIONS=(
    "GRU"
    "AdaCare"
)
DATASET_TASK_OPTIONS=(
    "tjh:outcome"
    "mimic-iv:outcome"
    "mimic-iv:readmission"
)

# Compute total runs for progress display
TOTAL_RUNS=$((${#DATASET_TASK_OPTIONS[@]}))
CURRENT_RUN=0

echo "Starting evaluation with ${TOTAL_RUNS} different configurations..."

# Iterate over dataset and task combinations
for DATASET_TASK in "${DATASET_TASK_OPTIONS[@]}"; do
    # Dataset and task
    IFS=":" read -r DATASET TASK <<< "$DATASET_TASK"

    # Add counter
    CURRENT_RUN=$((CURRENT_RUN + 1))

    # Construct command
    CMD="cd src/structured_ehr && python train_dl.py -d ${DATASET} -t ${TASK} -m ${MODEL_OPTIONS[@]}"

    # Print the counter
    echo "[$CURRENT_RUN/$TOTAL_RUNS] Running configuration..."
    echo "CMD: $CMD"
    echo "----------------------------------------"

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

echo "All training completed!"