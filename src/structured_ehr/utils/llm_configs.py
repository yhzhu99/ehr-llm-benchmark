import os

from dotenv import load_dotenv

load_dotenv()

LLM_MODELS_SETTINGS = {
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