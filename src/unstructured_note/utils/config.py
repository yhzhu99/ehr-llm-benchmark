"""
src/unstructured_note/utils/config.py
Configuration model file for the unstructured_note module
"""

MODELS_CONFIG = [
    {"model_type": "BERT", "model_name": "BERT", "hf_id": "bert-base-uncased"},
    {"model_type": "BERT", "model_name": "ClinicalBERT", "hf_id": "medicalai/ClinicalBERT"},
    {"model_type": "BERT", "model_name": "BioBERT", "hf_id": "pritamdeka/BioBert-PubMed200kRCT"},
    {"model_type": "BERT", "model_name": "GatorTron", "hf_id": "UFNLP/gatortron-base"},
    {"model_type": "BERT", "model_name": "Clinical-Longformer", "hf_id": "yikuan8/Clinical-Longformer"},
    
    ### medical LLM
    {"model_type": "GPT", "model_name": "BioGPT", "hf_id": "microsoft/biogpt"},
    {"model_type": "GPT", "model_name": "meditron", "hf_id": "epfl-llm/meditron-7b"},
    {"model_type": "GPT", "model_name": "OpenBioLLM", "hf_id": "aaditya/Llama3-OpenBioLLM-8B"},
    {"model_type": "GPT", "model_name": "BioMistral", "hf_id": "BioMistral/BioMistral-7B"},
    
    ### general LLM
    {"model_type": "GPT", "model_name": "GPT-2", "hf_id": "openai-community/gpt2"},
    {"model_type": "GPT", "model_name": "Qwen2.5-7B", "hf_id": "Qwen/Qwen2.5-7B"},
    {"model_type": "GPT", "model_name": "gemma-3-4b-pt", "hf_id": "google/gemma-3-4b-pt"},

    ### reasoning LLM
    {"model_type": "GPT", "model_name": "Llama-3.1-Nemotron-Nano-8B-v1", "hf_id": "nvidia/Llama-3.1-Nemotron-Nano-8B-v1"},
    {"model_type": "GPT", "model_name": "QwQ-32B", "hf_id": "qingcheng-ai/QWQ-32B-FP8"},
    {"model_type": "GPT", "model_name": "DeepSeek-R1-Distill-Qwen-7B", "hf_id": "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"},
]
