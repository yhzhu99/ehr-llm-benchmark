"""
src/unstructured_note/icd_code_clustering_task/get_icd_embeddings.py
Script to get ICD-10 embeddings for medical tasks
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

parser = argparse.ArgumentParser(description='Get ICD-10 embeddings')
parser.add_argument('--model', type=str, default='BERT', choices=BERTBasedModels + LLM)
parser.add_argument('--cuda', type=int, default=0, choices=[0,1,2,3,4,5,6,7])
parser.add_argument('--batch_size', type=int, default=1)
args = parser.parse_args()
model_name = args.model

class ICDDataset(Dataset):
    def __init__(self, dataset_path='my_datasets/icd10/icd10cm_order_2023.txt'):
        # Define split indices
        split_index = [6, 14, 16, 77]
        # Initialize empty list for data
        data = []

        # Open file and read data
        with open(dataset_path, 'r') as f:
            content = f.readline()
            while content:
                # Split content by indices and strip whitespace
                contents = [content[split_index[0]:split_index[1]].strip(),
                            content[split_index[1]:split_index[2]].strip(),
                            content[split_index[2]:split_index[3]].strip(),
                            content[split_index[3]:].strip()]
                code = contents[0]
                disease = contents[3]

                # Process codes less than or equal to 4 characters
                if len(code) <= 4:
                    data.append([code, disease])
                content = f.readline()

        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx][0], self.data[idx][1]  # code and disease name

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
    dataset = ICDDataset()
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    result = []
    for batch in tqdm(dataloader, desc=f'Processing ICD codes with {model_name}'):
        code, disease = batch

        # Tokenize inputs
        encoded_input = tokenizer(disease, padding="max_length", max_length=512, truncation=True, return_tensors="pt").to(device)

        with torch.no_grad():
            if model_name in LLM:
                outputs = model(**encoded_input, output_hidden_states=True)
                embedding = outputs.hidden_states[-1][0, -1, :].detach().cpu()
            else:
                outputs = model(**encoded_input)
                embedding = outputs.last_hidden_state[0, 0, :].detach().cpu()

        result.append({
            'code': code,
            'disease': disease,
            'embedding': embedding,
        })

    # Create output directory and save embeddings
    embedding_path = f"logs/icd/{model_name}/icd_embeddings.pkl"
    embedding_folder = Path(embedding_path).parent
    if not embedding_folder.exists():
        embedding_folder.mkdir(parents=True, exist_ok=True)

    pd.to_pickle(result, embedding_path)
    print(f"Saved embeddings to {embedding_path}")

if __name__ == '__main__':
    run()