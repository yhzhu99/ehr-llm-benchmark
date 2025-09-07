import os

from dotenv import load_dotenv

load_dotenv()

MODELS_CONFIG = {
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
        "model_name": "deepseek-reasoner",
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