#!/bin/bash

# Script to fine-tune GPT-based models for all tasks using IA3

# Define GPT models and tasks
GPT_MODELS=("BioGPT" "meditron" "OpenBioLLM" "BioMistral" "GPT-2" "Qwen2.5-7B" "gemma-3-4b-pt" "HuatuoGPT-o1-7B" "DeepSeek-R1-Distill-Qwen-7B")
TASKS=("mortality" "readmission")

# Fine-tune GPT models with IA3
for MODEL in "${GPT_MODELS[@]}"; do
    for TASK in "${TASKS[@]}"; do
        echo "Fine-tuning $MODEL for $TASK task using IA3 PEFT method"
        python src/unstructured_note/finetune_gpt_based_models/finetune_models.py --model "$MODEL" --task "$TASK" --batch_size 8 --learning_rate 8e-3 --epochs 5 --patience 1
    done
done

echo "All fine-tuning completed successfully!"