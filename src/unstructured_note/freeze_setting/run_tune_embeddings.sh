#!/bin/bash

# Script to train MLPs on frozen embeddings for all models on all tasks

# Define arrays of models and tasks
BERT_MODELS=("BERT" "ClinicalBERT" "BioBERT" "GatorTron" "Clinical-Longformer")
LLM_MODELS=("BioGPT" "meditron" "OpenBioLLM" "BioMistral" "GPT-2" "Qwen2.5-7B" "gemma-3-4b-pt" "HuatuoGPT-o1-7B" "DeepSeek-R1-Distill-Qwen-7B")
EMBEDDING_MODELS=("BGE-M3" "all-MiniLM-L6-v2" "BioBERT-embed" "BGE-Med")
TASKS=("mortality" "readmission")

# Train MLP on embeddings for all models
ALL_MODELS=("${BERT_MODELS[@]}" "${LLM_MODELS[@]}" "${EMBEDDING_MODELS[@]}")

for MODEL in "${ALL_MODELS[@]}"; do
    for TASK in "${TASKS[@]}"; do
        echo "Training MLP on $MODEL embeddings for $TASK task"
        python src/unstructured_note/freeze_setting/tune_embeddings.py --model "$MODEL" --task "$TASK" --batch_size 64 --learning_rate 1e-4 --epochs 50 --patience 5
    done
done

echo "All MLP trainings completed successfully!"