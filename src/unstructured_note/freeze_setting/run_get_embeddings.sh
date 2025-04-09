#!/bin/bash

# Script to generate embeddings for all models on all tasks

# Define arrays of models and tasks
BERT_MODELS=("BERT" "ClinicalBERT" "BioBERT" "GatorTron" "Clinical-Longformer")
LLM_MODELS=("BioGPT" "meditron" "OpenBioLLM" "BioMistral" "GPT-2" "Qwen2.5-7B" "gemma-3-4b-pt" "HuatuoGPT-o1-7B" "DeepSeek-R1-Distill-Qwen-7B")
TASKS=("mortality" "readmission")

# Generate embeddings for BERT models
for MODEL in "${BERT_MODELS[@]}"; do
    for TASK in "${TASKS[@]}"; do
        echo "Generating embeddings for $MODEL on $TASK task"
        python src/unstructured_note/freeze_setting/get_embeddings.py --model "$MODEL" --task "$TASK"
    done
done

# Generate embeddings for LLM models
for MODEL in "${LLM_MODELS[@]}"; do
    for TASK in "${TASKS[@]}"; do
        echo "Generating embeddings for $MODEL on $TASK task"
        python src/unstructured_note/freeze_setting/get_embeddings.py --model "$MODEL" --task "$TASK"
    done
done

echo "All embeddings generated successfully!"