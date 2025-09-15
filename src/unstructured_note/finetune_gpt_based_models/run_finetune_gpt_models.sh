#!/bin/bash

# Script to fine-tune GPT-based models for all tasks using LoRA

# Define GPT models and tasks
GPT_MODELS=("BioGPT" "meditron" "OpenBioLLM" "BioMistral" "GPT-2" "Qwen2.5-7B" "gemma-3-4b-pt" "HuatuoGPT-o1-7B" "DeepSeek-R1-Distill-Qwen-7B")
DATASET_TASK_OPTIONS=("mimic-iv:mortality" "mimic-iv:readmission" "mimic-iii:mortality")

# Fine-tune GPT models with LoRA
for MODEL in "${GPT_MODELS[@]}"; do
    for DATASET_TASK in "${DATASET_TASK_OPTIONS[@]}"; do
        IFS=":" read -r DATASET TASK <<< "$DATASET_TASK"
        echo "Fine-tuning $MODEL for $TASK task on $DATASET dataset using LoRA PEFT method"
        python src/unstructured_note/finetune_gpt_based_models/finetune_models.py \
            --model "$MODEL" \
            --task "$TASK" \
            --dataset "$DATASET" \
            --batch_size 8 \
            --learning_rate 2e-4 \
            --lora_rank 8 \
            --lora_alpha 16 \
            --lora_dropout 0.1 \
            --epochs 5 \
            --patience 1
    done
done

echo "All fine-tuning completed successfully!"