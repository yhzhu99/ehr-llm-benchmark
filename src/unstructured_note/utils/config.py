"""
src/unstructured_note/utils/config.py
Configuration model file for the unstructured_note module
"""
import os

from dotenv import load_dotenv

load_dotenv()

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
    {"model_type": "GPT", "model_name": "HuatuoGPT-o1-7B", "hf_id": "FreedomIntelligence/HuatuoGPT-o1-7B"},
    {"model_type": "GPT", "model_name": "DeepSeek-R1-Distill-Qwen-7B", "hf_id": "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"},

    ### embedding models
    {"model_type": "embedding", "model_name": "BGE-M3", "hf_id": "BAAI/bge-m3"},
    {"model_type": "embedding", "model_name": "all-MiniLM-L6-v2", "hf_id": "sentence-transformers/all-MiniLM-L6-v2"},
    {"model_type": "embedding", "model_name": "BioBERT-embed", "hf_id": "pritamdeka/BioBERT-mnli-snli-scinli-scitail-mednli-stsb"},
    {"model_type": "embedding", "model_name": "BGE-Med", "hf_id": "ls-da3m0ns/bge_large_medical"},
]

LLM_API_CONFIG = {
    "deepseek-v3-official": {
        "api_key": os.getenv("DEEPSEEK_API_KEY"),
        "base_url": "https://api.deepseek.com",
        "model_name": "deepseek-chat",
        "comment": "DeepSeek V3 Official",
        "reasoning": False,
    },
    "deepseek-r1-official": {
        "api_key": os.getenv("DEEPSEEK_API_KEY"),
        "base_url": "https://api.deepseek.com",
        "model_name": "deepseek-reasoner",
        "comment": "DeepSeek R1 Reasoning Model Official",
        "reasoning": True,
    },
    "deepseek-v3-ali": {
        "api_key": os.getenv("DASHSCOPE_API_KEY"),
        "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "model_name": "deepseek-v3",
        "comment": "DeepSeek V3 Ali",
        "reasoning": False,
    },
    "deepseek-r1-ali": {
        "api_key": os.getenv("DASHSCOPE_API_KEY"),
        "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "model_name": "deepseek-r1",
        "comment": "DeepSeek R1 Reasoning Model Ali",
        "reasoning": True,
    },
    "deepseek-v3-ark": {
        "api_key": os.getenv("ARK_API_KEY"),
        "base_url": "https://ark.cn-beijing.volces.com/api/v3",
        "model_name": "deepseek-v3-250324",
        "comment": "DeepSeek V3 Ark",
        "reasoning": False,
    },
    "deepseek-r1-ark": {
        "api_key": os.getenv("ARK_API_KEY"),
        "base_url": "https://ark.cn-beijing.volces.com/api/v3",
        "model_name": "deepseek-r1-250120",
        "comment": "DeepSeek R1 Reasoning Model Ark",
        "reasoning": True,
    },
}