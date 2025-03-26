#!/bin/bash

# Script to run sentence embedding generation for all models on the biosses dataset

# Define CUDA devices to use (can cycle through if you have multiple GPUs)
CUDA_DEVICES=(0 1 2 3)  # Adjust based on available GPUs

# Counter to track current model for GPU assignment
model_counter=0

# Import models from config
# This script assumes that the models are defined in MODELS_CONFIG in the config.py file
models=$(python -c "
import sys
sys.path.append('src/unstructured_note/utils')
from config import MODELS_CONFIG
for model in MODELS_CONFIG:
    print(model['model_name'])
")

# Loop through all models and run the embedding generation
for model_name in $models; do
    echo "Running $model_name model"
    python src/unstructured_note/medical_sentence_matching_task/get_sentence_embeddings.py \
        --model="$model_name" \
        --cuda=${CUDA_DEVICES[$model_counter % ${#CUDA_DEVICES[@]}]}
    model_counter=$((model_counter+1))
done

echo "All sentence embedding generation tasks completed!"
