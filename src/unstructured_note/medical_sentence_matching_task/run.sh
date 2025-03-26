#!/bin/bash

# Script to run sentence embedding generation for all models on the biosses dataset

# Define CUDA devices to use (can cycle through if you have multiple GPUs)
CUDA_DEVICES=(0 1 2 3)  # Adjust based on available GPUs

# Counter to track current model for GPU assignment
model_counter=0

# BERT models
echo "Running BERT model"
python src/unstructured_note/medical_sentence_matching_task/get_sentence_embeddings.py --model="BERT" --cuda=${CUDA_DEVICES[$model_counter % ${#CUDA_DEVICES[@]}]}
model_counter=$((model_counter+1))

echo "Running ClinicalBERT model"
python src/unstructured_note/medical_sentence_matching_task/get_sentence_embeddings.py --model="ClinicalBERT" --cuda=${CUDA_DEVICES[$model_counter % ${#CUDA_DEVICES[@]}]}
model_counter=$((model_counter+1))

echo "Running BioBERT model"
python src/unstructured_note/medical_sentence_matching_task/get_sentence_embeddings.py --model="BioBERT" --cuda=${CUDA_DEVICES[$model_counter % ${#CUDA_DEVICES[@]}]}
model_counter=$((model_counter+1))

echo "Running GatorTron model"
python src/unstructured_note/medical_sentence_matching_task/get_sentence_embeddings.py --model="GatorTron" --cuda=${CUDA_DEVICES[$model_counter % ${#CUDA_DEVICES[@]}]}
model_counter=$((model_counter+1))

echo "Running Clinical-Longformer model"
python src/unstructured_note/medical_sentence_matching_task/get_sentence_embeddings.py --model="Clinical-Longformer" --cuda=${CUDA_DEVICES[$model_counter % ${#CUDA_DEVICES[@]}]}
model_counter=$((model_counter+1))

# Medical LLMs
echo "Running BioGPT model"
python src/unstructured_note/medical_sentence_matching_task/get_sentence_embeddings.py --model="BioGPT" --cuda=${CUDA_DEVICES[$model_counter % ${#CUDA_DEVICES[@]}]}
model_counter=$((model_counter+1))

echo "Running MedAlpaca model"
python src/unstructured_note/medical_sentence_matching_task/get_sentence_embeddings.py --model="MedAlpaca" --cuda=${CUDA_DEVICES[$model_counter % ${#CUDA_DEVICES[@]}]}
model_counter=$((model_counter+1))

echo "Running HuatuoGPT model"
python src/unstructured_note/medical_sentence_matching_task/get_sentence_embeddings.py --model="HuatuoGPT" --cuda=${CUDA_DEVICES[$model_counter % ${#CUDA_DEVICES[@]}]}
model_counter=$((model_counter+1))

echo "Running meditron model"
python src/unstructured_note/medical_sentence_matching_task/get_sentence_embeddings.py --model="meditron" --cuda=${CUDA_DEVICES[$model_counter % ${#CUDA_DEVICES[@]}]}
model_counter=$((model_counter+1))

echo "Running OpenBioLLM model"
python src/unstructured_note/medical_sentence_matching_task/get_sentence_embeddings.py --model="OpenBioLLM" --cuda=${CUDA_DEVICES[$model_counter % ${#CUDA_DEVICES[@]}]}
model_counter=$((model_counter+1))

echo "Running BioMistral model"
python src/unstructured_note/medical_sentence_matching_task/get_sentence_embeddings.py --model="BioMistral" --cuda=${CUDA_DEVICES[$model_counter % ${#CUDA_DEVICES[@]}]}
model_counter=$((model_counter+1))

echo "Running Baichuan-M1 model"
python src/unstructured_note/medical_sentence_matching_task/get_sentence_embeddings.py --model="Baichuan-M1" --cuda=${CUDA_DEVICES[$model_counter % ${#CUDA_DEVICES[@]}]}
model_counter=$((model_counter+1))

# General LLMs
echo "Running GPT-2 model"
python src/unstructured_note/medical_sentence_matching_task/get_sentence_embeddings.py --model="GPT-2" --cuda=${CUDA_DEVICES[$model_counter % ${#CUDA_DEVICES[@]}]}
model_counter=$((model_counter+1))

echo "Running Qwen2.5-7B model"
python src/unstructured_note/medical_sentence_matching_task/get_sentence_embeddings.py --model="Qwen2.5-7B" --cuda=${CUDA_DEVICES[$model_counter % ${#CUDA_DEVICES[@]}]}
model_counter=$((model_counter+1))

echo "Running gemma-3-4b-pt model"
python src/unstructured_note/medical_sentence_matching_task/get_sentence_embeddings.py --model="gemma-3-4b-pt" --cuda=${CUDA_DEVICES[$model_counter % ${#CUDA_DEVICES[@]}]}
model_counter=$((model_counter+1))

# Reasoning LLMs
echo "Running Llama-3.1-Nemotron-Nano-8B-v1 model"
python src/unstructured_note/medical_sentence_matching_task/get_sentence_embeddings.py --model="Llama-3.1-Nemotron-Nano-8B-v1" --cuda=${CUDA_DEVICES[$model_counter % ${#CUDA_DEVICES[@]}]}
model_counter=$((model_counter+1))

echo "Running QwQ-32B model"
python src/unstructured_note/medical_sentence_matching_task/get_sentence_embeddings.py --model="QwQ-32B" --cuda=${CUDA_DEVICES[$model_counter % ${#CUDA_DEVICES[@]}]}
model_counter=$((model_counter+1))

echo "Running DeepSeek-R1-Distill-Qwen-7B model"
python src/unstructured_note/medical_sentence_matching_task/get_sentence_embeddings.py --model="DeepSeek-R1-Distill-Qwen-7B" --cuda=${CUDA_DEVICES[$model_counter % ${#CUDA_DEVICES[@]}]}
model_counter=$((model_counter+1))

echo "All sentence embedding generation tasks completed!"