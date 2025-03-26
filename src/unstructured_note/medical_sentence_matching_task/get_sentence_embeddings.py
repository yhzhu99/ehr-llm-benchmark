"""
src/unstructured_note/medical_sentence_matching_task/get_sentence_embeddings.py
Script to get sentence embeddings for medical sentence similarity task
"""

from pathlib import Path
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
import argparse

import torch
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM, set_seed
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from tqdm import tqdm

from unstructured_note.utils.config import MODELS_CONFIG

# Check if MPS is available (for Mac GPU)
mps_available = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
set_seed(42)

# Create model lists for argument selection
BERTBasedModels = [model["model_name"] for model in MODELS_CONFIG if model["model_type"] == "BERT"]
LLM = [model["model_name"] for model in MODELS_CONFIG if model["model_type"] == "GPT"]

parser = argparse.ArgumentParser(description='Get sentence embeddings for biosses dataset')
parser.add_argument('--model', type=str, default='BERT', choices=BERTBasedModels + LLM)
parser.add_argument('--cuda', type=int, default=0, choices=[0,1,2,3,4,5,6,7])
parser.add_argument('--batch_size', type=int, default=1)
args = parser.parse_args()
model_name = args.model

class BIOSSESDataset(Dataset):
    def __init__(self) -> None:
        self.data = pd.read_parquet("my_datasets/biosses/train-00000-of-00001.parquet")

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        return {
            'sentence1': row['sentence1'],
            'sentence2': row['sentence2'],
            'score': row['score']
        }

def run():
    # Get model path from config
    model_path = next(model["hf_id"] for model in MODELS_CONFIG if model["model_name"] == model_name)
    
    # Determine the appropriate device
    if mps_available:
        device = torch.device("mps")
        print("Using Mac GPU (MPS)")
    elif torch.cuda.is_available():
        device = torch.device(f'cuda:{args.cuda}')
        print(f"Using CUDA device {args.cuda}")
    else:
        device = torch.device("cpu")
        print("Using CPU")
    
    # Load model based on type
    if model_name in LLM:
        model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True).to(device)
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = tokenizer.eos_token_id
    elif model_name in BERTBasedModels:
        model = AutoModel.from_pretrained(model_path, trust_remote_code=True).to(device)
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    else:
        raise ValueError(f"Model {model_name} not supported.")
    
    # Set up data loader
    dataloader = DataLoader(BIOSSESDataset(), batch_size=args.batch_size, shuffle=False)
    
    result = []
    for batch in tqdm(dataloader, desc=f'Processing sentences with {model_name}'):
        sentence1, sentence2, score = batch['sentence1'], batch['sentence2'], batch['score']
        
        # Tokenize inputs
        encoded_input1 = tokenizer(sentence1, padding="max_length", max_length=512, truncation=True, return_tensors="pt").to(device)
        encoded_input2 = tokenizer(sentence2, padding="max_length", max_length=512, truncation=True, return_tensors="pt").to(device)
        
        with torch.no_grad():
            if model_name in LLM:
                outputs1 = model(**encoded_input1, output_hidden_states=True)
                outputs2 = model(**encoded_input2, output_hidden_states=True)
                embedding1 = outputs1.hidden_states[-1][0, -1, :].detach().cpu()
                embedding2 = outputs2.hidden_states[-1][0, -1, :].detach().cpu()
            else:
                outputs1 = model(**encoded_input1)
                outputs2 = model(**encoded_input2)
                embedding1 = outputs1.last_hidden_state[0, 0, :].detach().cpu()
                embedding2 = outputs2.last_hidden_state[0, 0, :].detach().cpu()
                
        result.append({
            'embedding1': embedding1,
            'embedding2': embedding2,
            'score': score.item()
        })
    
    # Create output directory and save embeddings
    embedding_path = f"logs/biosses/{model_name}/sentence_embeddings.pkl"
    embedding_folder = Path(embedding_path).parent
    if not embedding_folder.exists():
        embedding_folder.mkdir(parents=True, exist_ok=True)
    
    pd.to_pickle(result, embedding_path)
    print(f"Saved embeddings to {embedding_path}")

if __name__ == '__main__':
    run()