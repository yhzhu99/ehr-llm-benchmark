#!/bin/bash

# Script to fine-tune BERT-based models for all tasks

# Define BERT models and tasks
BERT_MODELS=("BERT" "ClinicalBERT" "BioBERT" "GatorTron" "Clinical-Longformer")
DATASET_TASK_OPTIONS=("mimic-iv:mortality" "mimic-iv:readmission" "mimic-iii:mortality")

# Fine-tune BERT models
for MODEL in "${BERT_MODELS[@]}"; do
    for DATASET_TASK in "${DATASET_TASK_OPTIONS[@]}"; do
        IFS=":" read -r DATASET TASK <<< "$DATASET_TASK"
        echo "Fine-tuning $MODEL for $TASK task on $DATASET dataset"
        python src/unstructured_note/finetune_bert_based_models/finetune_models.py --model "$MODEL" --task "$TASK" --dataset "$DATASET" --batch_size 16 --learning_rate 1e-5 --epochs 10 --patience 3
    done
done

echo "All fine-tuning completed successfully!"