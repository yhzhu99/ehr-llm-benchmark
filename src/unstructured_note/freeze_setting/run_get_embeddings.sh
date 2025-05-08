#!/bin/bash

# Script to generate embeddings for all models on all tasks

# Define arrays of models and tasks
BERT_MODELS=("BERT" "ClinicalBERT" "BioBERT" "GatorTron" "Clinical-Longformer")
LLM_MODELS=("BioGPT" "meditron" "OpenBioLLM" "BioMistral" "GPT-2" "Qwen2.5-7B" "gemma-3-4b-pt" "HuatuoGPT-o1-7B" "DeepSeek-R1-Distill-Qwen-7B")
EMBEDDING_MODELS=("BGE-M3" "all-MiniLM-L6-v2" "BioBERT-embed" "BGE-Med")

# Generate embeddings for BERT models
for MODEL in "${BERT_MODELS[@]}"; do
    echo "Generating embeddings for $MODEL"
    python src/unstructured_note/freeze_setting/get_embeddings.py --model $MODEL
done

# Generate embeddings for LLM models
for MODEL in "${LLM_MODELS[@]}"; do
    echo "Generating embeddings for $MODEL"
    python src/unstructured_note/freeze_setting/get_embeddings.py --model $MODEL
done

# Generate embeddings for Embedding models
for MODEL in "${EMBEDDING_MODELS[@]}"; do
    echo "Generating embeddings for $MODEL"
    python src/unstructured_note/freeze_setting/get_embeddings.py --model $MODEL
done

echo "All embeddings generated successfully!"