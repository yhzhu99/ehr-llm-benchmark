#!/bin/bash

# Script to fine-tune BERT-based models for all tasks

# Define BERT models and tasks
BERT_MODELS=("BERT" "ClinicalBERT" "BioBERT" "GatorTron" "Clinical-Longformer")
TASKS=("mortality" "readmission")

# Fine-tune BERT models
for MODEL in "${BERT_MODELS[@]}"; do
    for TASK in "${TASKS[@]}"; do
        echo "Fine-tuning $MODEL for $TASK task"
        python src/unstructured_note/finetune_bert_based_models/finetune_models.py --model "$MODEL" --task "$TASK" --batch_size 16 --learning_rate 1e-5 --epochs 10 --patience 3
    done
done

echo "All fine-tuning completed successfully!"