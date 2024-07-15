import os
import ssl
ssl._create_default_https_context=ssl._create_unverified_context
from huggingface_hub import snapshot_download

from utils.config import LLM, BERTBasedModels, RepoIDs, TOKENForHF

save_dir = "./HF_models"
os.makedirs(save_dir, exist_ok=True)

for repo_id, model in zip(RepoIDs, BERTBasedModels + LLM):
    snapshot_download(repo_id=repo_id, cache_dir=f"./HF_models/{model}", token=TOKENForHF)