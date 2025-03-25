#!/bin/bash

# Script to run embeddings generation for all models in config on both datasets

# Define datasets
DATASETS=("mortality" "discharge")

# Loop through each model in the config and run the command
for DATASET in "${DATASETS[@]}"; do
    # BERT models
    echo "Running BERT model on ${DATASET} dataset"
    python src/unstructured_note/freeze_setting/get_embeddings.py --model="BERT" --dataset="${DATASET}"
    
    echo "Running ClinicalBERT model on ${DATASET} dataset"
    python src/unstructured_note/freeze_setting/get_embeddings.py --model="ClinicalBERT" --dataset="${DATASET}"
    
    echo "Running BioBERT model on ${DATASET} dataset"
    python src/unstructured_note/freeze_setting/get_embeddings.py --model="BioBERT" --dataset="${DATASET}"
    
    echo "Running GatorTron model on ${DATASET} dataset"
    python src/unstructured_note/freeze_setting/get_embeddings.py --model="GatorTron" --dataset="${DATASET}"
    
    echo "Running Clinical-Longformer model on ${DATASET} dataset"
    python src/unstructured_note/freeze_setting/get_embeddings.py --model="Clinical-Longformer" --dataset="${DATASET}"
    
    # Medical LLMs
    echo "Running BioGPT model on ${DATASET} dataset"
    python src/unstructured_note/freeze_setting/get_embeddings.py --model="BioGPT" --dataset="${DATASET}"
    
    echo "Running MedAlpaca model on ${DATASET} dataset"
    python src/unstructured_note/freeze_setting/get_embeddings.py --model="MedAlpaca" --dataset="${DATASET}"
    
    echo "Running HuatuoGPT model on ${DATASET} dataset"
    python src/unstructured_note/freeze_setting/get_embeddings.py --model="HuatuoGPT" --dataset="${DATASET}"
    
    echo "Running meditron model on ${DATASET} dataset"
    python src/unstructured_note/freeze_setting/get_embeddings.py --model="meditron" --dataset="${DATASET}"
    
    echo "Running OpenBioLLM model on ${DATASET} dataset"
    python src/unstructured_note/freeze_setting/get_embeddings.py --model="OpenBioLLM" --dataset="${DATASET}"
    
    echo "Running BioMistral model on ${DATASET} dataset"
    python src/unstructured_note/freeze_setting/get_embeddings.py --model="BioMistral" --dataset="${DATASET}"
    
    echo "Running Baichuan-M1 model on ${DATASET} dataset"
    python src/unstructured_note/freeze_setting/get_embeddings.py --model="Baichuan-M1" --dataset="${DATASET}"
    
    # General LLMs
    echo "Running GPT-2 model on ${DATASET} dataset"
    python src/unstructured_note/freeze_setting/get_embeddings.py --model="GPT-2" --dataset="${DATASET}"
    
    echo "Running Llama3 model on ${DATASET} dataset"
    python src/unstructured_note/freeze_setting/get_embeddings.py --model="Llama3" --dataset="${DATASET}"
    
    echo "Running Qwen2.5-7B model on ${DATASET} dataset"
    python src/unstructured_note/freeze_setting/get_embeddings.py --model="Qwen2.5-7B" --dataset="${DATASET}"
    
    # Reasoning LLMs
    echo "Running QwQ-32B model on ${DATASET} dataset"
    python src/unstructured_note/freeze_setting/get_embeddings.py --model="QwQ-32B" --dataset="${DATASET}"
    
    echo "Running DeepSeek-R1-Distill-Qwen-7B model on ${DATASET} dataset"
    python src/unstructured_note/freeze_setting/get_embeddings.py --model="DeepSeek-R1-Distill-Qwen-7B" --dataset="${DATASET}"
done

echo "All embeddings generation tasks completed!"