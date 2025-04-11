#!/bin/bash

# Script to run sentence embedding generation for all models on the biosses dataset

# Define CUDA devices to use (can cycle through if you have multiple GPUs)
CUDA_DEVICES=(0 1 2 3)  # Adjust based on available GPUs

# Counter to track current model for GPU assignment
model_counter=0

# Loop through all models and run the embedding generation
for model in BERT ClinicalBERT BioBERT GatorTron Clinical-Longformer BioGPT meditron OpenBioLLM BioMistral GPT-2 Qwen2.5-7B gemma-3-4b-pt HuatuoGPT-o1-7B DeepSeek-R1-Distill-Qwen-7B BGE-M3 all-MiniLM-L6-v2 BioBERT-embed BGE-Med; do
    echo "Running $model model"
    python src/unstructured_note/medical_sentence_matching_task/get_sentence_embeddings.py \
        --model="$model" \
        --cuda=${CUDA_DEVICES[$model_counter % ${#CUDA_DEVICES[@]}]}
    model_counter=$((model_counter+1))
done

echo "All sentence embedding generation tasks completed!"