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
    "llm-studio": {
        "api_key": os.getenv("LLMSTUDIO_API_KEY"),
        "base_url": "https://llm.yhzhu.uk/v1"
    },
    "v8": {
        "api_key": os.getenv("XDAI_V8_API_KEY"),
        "base_url": "https://xdaicn.top/v1",
    },
    "default": {
        "api_key": os.getenv("XDAI_DEFAULT_API_KEY"),
        "base_url": "https://xdaicn.top/v1",
    },
}