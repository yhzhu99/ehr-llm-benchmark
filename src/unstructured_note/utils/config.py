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

LMSTUDIO_MODELS_CONFIG = {
    ### medical LLM
    "OpenBioLLM": {"model_type": "GPT", "model_name": "OpenBioLLM", "lmstudio_id": "openbiollm-llama3-8b"},

    ### general LLM
    "GPT-2": {"model_type": "GPT", "model_name": "GPT-2", "hf_id": "openai-community/gpt2"},
    "Qwen2.5-7B": {"model_type": "GPT", "model_name": "Qwen2.5-7B", "lmstudio_id": "qwen2.5-7b-instruct-1m"},
    "Gemma-3-4B": {"model_type": "GPT", "model_name": "gemma-3-4b-pt", "lmstudio_id": "gemma-3-4b-it"},

    ### reasoning LLM
    "HuatuoGPT-o1-7B": {"model_type": "GPT", "model_name": "HuatuoGPT-o1-7B", "lmstudio_id": "huatuogpt-o1-7b"},
    "DeepSeek-R1-7B": {"model_type": "GPT", "model_name": "DeepSeek-R1-Distill-Qwen-7B", "lmstudio_id": "deepseek-r1-distill-qwen-7b"},
}

LLM_API_CONFIG = {
    "deepseek-v3-chat": {
        "api_key": os.getenv("DEEPSEEK_API_KEY"),
        "base_url": "https://api.deepseek.com/v1",
        "model_name": "deepseek-chat",
        "comment": "DeepSeek V3 Chat",
    },
    "deepseek-v3-reasoner": {
        "api_key": os.getenv("DEEPSEEK_API_KEY"),
        "base_url": "https://api.deepseek.com/v1",
        "model_name": "deepseek-v3-reasoner",
        "comment": "DeepSeek V3 Reasoner",
    },
    "llm-studio": {
        "api_key": os.getenv("LLMSTUDIO_API_KEY"),
        "base_url": "https://llm.yhzhu.uk/v1"
    },
    "laozhang": {
        "api_key": os.getenv("LAOZHANG_API_KEY"),
        "base_url": "https://api.laozhang.ai/v1"
    }
}