BERTBasedModels = ["BERT", "ClinicalBERT", "BioBERT", "Clinical-Longformer", "GatorTron"]
LLM = ["GPT-2", "BioGPT", "MedAlpaca", "HuatuoGPT", "meditron", "OpenBioLLM", "Llama3"]

LLMPathList = {"GPT-2": "HF_models/GPT-2",
    "BioGPT": "HF_models/BioGPT",
    "MedAlpaca": "HF_models/MedAlpaca/models--medalpaca--medalpaca-7b/snapshots/fbb41b75d5a46ba405d496db1083a6f1d3df72a2",
    "HuatuoGPT": "HF_models/HuatuoGPT/models--FreedomIntelligence--HuatuoGPT2-7B/snapshots/be622086b8e60326c9b3c193de72194e006f46cc",
    "meditron": "HF_models/meditron/models--epfl-llm--meditron-7b/snapshots/d7d0a5ed929384a6b059ac74198cf1d71f44ba76",
    "OpenBioLLM": "HF_models/OpenBioLLM/models--aaditya--Llama3-OpenBioLLM-8B/snapshots/000c725dc3a680e35260b2c213163387581c974f",
    "Llama3": "HF_models/Meta-Llama-3-8B"}

RepoIDs = ["bert-base-uncased", "medicalai/ClinicalBERT", "pritamdeka/BioBert-PubMed200kRCT", "UFNLP/gatortron-base", "yikuan8/Clinical-Longformer"] + \
    ["openai-community/gpt2", "microsoft/biogpt", "medalpaca/medalpaca-7b", "FreedomIntelligence/HuatuoGPT2-7B", "epfl-llm/meditron-7b", "aaditya/Llama3-OpenBioLLM-8B", "meta-llama/Meta-Llama-3-8B-Instruct"]
    
TOKENForHF = ""
