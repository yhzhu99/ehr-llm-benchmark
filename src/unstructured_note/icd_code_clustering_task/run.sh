#!/bin/bash

# Loop through all models
for model in BERT ClinicalBERT BioBERT GatorTron Clinical-Longformer BioGPT meditron OpenBioLLM BioMistral GPT-2 Qwen2.5-7B gemma-3-4b-pt HuatuoGPT-o1-7B DeepSeek-R1-Distill-Qwen-7B BGE-M3 all-MiniLM-L6-v2 BioBERT-embed BGE-Med; do
  python src/unstructured_note/icd_code_clustering_task/get_icd_embeddings.py --model $model
done

python src/unstructured_note/icd_code_clustering_task/process_icd_results.py --process_all