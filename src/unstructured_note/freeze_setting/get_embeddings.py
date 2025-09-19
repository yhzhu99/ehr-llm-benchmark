"""
src/unstructured_note/freeze_setting/get_embeddings.py
This script is used to get embeddings from the model for the textual data in the MIMIC-III or MIMIC-IV dataset.
"""

from pathlib import Path
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["TOKENIZERS_PARALLELISM"] = "false"  # Add this line at the top

import argparse

from tqdm import tqdm
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModel, set_seed

# Import libraries for new embedding models
from sentence_transformers import SentenceTransformer
from FlagEmbedding import BGEM3FlagModel

from src.unstructured_note.utils.config import MODELS_CONFIG

# Set seed for reproducibility
set_seed(42)

# Check if MPS is available (for Mac GPU)
mps_available = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()

# Create model type lists
BERT_MODELS = [model["model_name"] for model in MODELS_CONFIG if model["model_type"] == "BERT"]
LLM_MODELS = [model["model_name"] for model in MODELS_CONFIG if model["model_type"] == "GPT"]

parser = argparse.ArgumentParser(description='Generate embeddings from models')
parser.add_argument('--model', type=str, required=True, choices=BERT_MODELS + LLM_MODELS)
parser.add_argument('--dataset', type=str, required=True, choices=["mimic-iv", "mimic-iii"])
parser.add_argument('--batch_size', type=int, default=1)
parser.add_argument('--max_length', type=int, default=512)
args = parser.parse_args()

# Get model info
model_info = next(model for model in MODELS_CONFIG if model["model_name"] == args.model)
model_path = model_info["hf_id"]
model_type = model_info["model_type"]

# Determine device
if torch.cuda.is_available():
    device = torch.device("cuda")
    print(f"Using CUDA")
elif mps_available:
    device = torch.device("mps")
    print("Using Mac GPU (MPS)")
else:
    device = torch.device("cpu")
    print("Using CPU")

# Load tokenizer and model
print(f"Loading model: {args.model} from {model_path}")
if model_type == "GPT":
    model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = tokenizer.eos_token_id
elif model_type == "BERT":
    model = AutoModel.from_pretrained(model_path, trust_remote_code=True).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
else:
    raise ValueError(f"Model type {model_type} not supported.")

# Load data
data_splits = {}
for split in ['train', 'val', 'test']:
    split_name = f'{split}_data.pkl'
    file_path = f"my_datasets/{args.dataset}/processed/split/{split_name}"
    data_splits[split] = pd.read_pickle(file_path)
    print(f"Loaded {len(data_splits[split])} records from {file_path}")

def extract_embeddings(data_split, split_name):
    """Extract embeddings for a data split"""
    print(f"Extracting embeddings for {split_name} split")
    embeddings = []

    for item in tqdm(data_split, desc=f"Processing {split_name}"):
        text = item['x_note']
        label_mortality = item['y_mortality']
        label_readmission = item['y_readmission']

        # Tokenize for BERT or LLM models
        inputs = tokenizer(
            text,
            return_tensors="pt",
            padding="max_length",
            max_length=args.max_length,
            truncation=True
        ).to(device)

        # Get embedding
        with torch.no_grad():
            if model_type == "GPT":
                outputs = model(**inputs, output_hidden_states=True)
                # For GPT models, use last hidden state of the last token
                last_hidden_state = outputs.hidden_states[-1]
                embedding = last_hidden_state[0, -1, :].detach().cpu()
            else:  # BERT models
                outputs = model(**inputs)
                # For BERT models, use the [CLS] token embedding
                embedding = outputs.last_hidden_state[0, 0, :].detach().cpu()

        # Store embedding with label
        embedding_dict = {
            'embedding': embedding,
            'y_mortality': label_mortality,
            'y_readmission': label_readmission,
        }

        embeddings.append(embedding_dict)

    return embeddings

# Create output directory
output_dir = Path(f"logs/unstructured_note/{args.dataset}-note/embeddings/{args.model}")
output_dir.mkdir(parents=True, exist_ok=True)

# Extract and save embeddings for each split
for split, split_data in data_splits.items():
    split_name = split
    output_path = output_dir / f"{split_name}_embeddings.pkl"
    if os.path.exists(output_path):
        print(f"Embeddings already exist at {output_path}")
        continue

    embeddings = extract_embeddings(split_data, split_name)
    pd.to_pickle(embeddings, output_path)
    print(f"Saved {len(embeddings)} embeddings to {output_path}")

print("Embedding extraction complete!")