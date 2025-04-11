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

# Import libraries for new embedding models
from sentence_transformers import SentenceTransformer
from FlagEmbedding import BGEM3FlagModel

from unstructured_note.utils.config import MODELS_CONFIG

# Check if MPS is available (for Mac GPU)
mps_available = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
set_seed(42)

# Create model lists for argument selection
BERT_MODELS = [model["model_name"] for model in MODELS_CONFIG if model["model_type"] == "BERT"]
LLM_MODELS = [model["model_name"] for model in MODELS_CONFIG if model["model_type"] == "GPT"]
EMBEDDING_MODELS = [model["model_name"] for model in MODELS_CONFIG if model["model_type"] == "embedding"]

parser = argparse.ArgumentParser(description='Get ICD-10 embeddings')
parser.add_argument('--model', type=str, default='BERT', choices=BERT_MODELS + LLM_MODELS + EMBEDDING_MODELS)
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
    # Get model info from config
    model_info = next(model for model in MODELS_CONFIG if model["model_name"] == model_name)
    model_path = model_info["hf_id"]
    model_type = model_info["model_type"]

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
    if model_type == "GPT":
        model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True).to(device)
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = tokenizer.eos_token_id
    elif model_type == "BERT":
        model = AutoModel.from_pretrained(model_path, trust_remote_code=True).to(device)
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    elif model_type == "embedding":
        if model_name == "BGE-M3":
            model = BGEM3FlagModel(model_path, use_fp16=True)
            tokenizer = None
        else:
            # SentenceTransformer models
            model = SentenceTransformer(model_path)
            tokenizer = None
    else:
        raise ValueError(f"Model type {model_type} not supported.")

    # Set up data loader
    dataset = ICDDataset()
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    result = []
    for batch in tqdm(dataloader, desc=f'Processing ICD codes with {model_name}'):
        code, disease = batch

        if model_type == "embedding":
            # Process with embedding models
            if model_name == "BGE-M3":
                # BGE-M3 model
                outputs = model.encode(disease)['dense_vecs']
                embedding = torch.tensor(outputs[0])
            else:
                # SentenceTransformer models
                embedding = torch.tensor(model.encode(disease)[0])
        else:
            # Process with BERT or LLM models
            encoded_input = tokenizer(disease, padding="max_length", max_length=512, truncation=True, return_tensors="pt").to(device)

            with torch.no_grad():
                if model_type == "GPT":
                    outputs = model(**encoded_input, output_hidden_states=True)
                    embedding = outputs.hidden_states[-1][0, -1, :].detach().cpu()
                else:  # BERT models
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