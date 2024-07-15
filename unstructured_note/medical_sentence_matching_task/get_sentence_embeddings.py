import os

import torch
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from datasets import load_dataset
from tqdm import tqdm
from accelerate import Accelerator

from utils.config import BERTBasedModels, LLM, LLMPathList


class MyDataset(Dataset):
    def __init__(self, dataset_name, cache_dir='datasets') -> None:
        self.data = load_dataset(dataset_name, cache_dir=cache_dir)['train']

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]

# accelerator = Accelerator()
# device = accelerator.device
# device = torch.device('cuda:1') if torch.cuda.is_available() else torch.device("cpu") 

def run(model_name):
    dataset_name = 'tabilab/biosses'
    batch_size = 1
    save_dir = 'logs/similarity'   
    dataloader = DataLoader(MyDataset(dataset_name), batch_size=batch_size, shuffle=False)
    
    if model_name in LLM:
        accelerator = Accelerator()
        device = accelerator.device
        
        model_path = LLMPathList[model_name]
        if model_name != 'BioGPT':
            model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True, device_map="auto")
            model = accelerator.prepare(model)
        else:
            model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True).to(device)
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = tokenizer.eos_token_id
    else:
        device = torch.device('cuda:1') if torch.cuda.is_available() else torch.device("cpu")
        
        model_path = f'HF_models/{model_name}'
        model = AutoModel.from_pretrained(model_path, trust_remote_code=True).to(device)
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    
    result = []
    for batch in tqdm(dataloader, desc=f'Running sentence task for model: {model_name}'):
        sentence1, sentence2, score = batch['sentence1'], batch['sentence2'], batch['score']
        encoded_input1 = tokenizer(sentence1, padding="max_length", max_length=512, truncation=True, return_tensors="pt").to(device)
        encoded_input2 = tokenizer(sentence2, padding="max_length", max_length=512, truncation=True, return_tensors="pt").to(device)
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
            'score': score
        })
    save_path = os.path.join(save_dir, model_name)
    os.makedirs(save_path, exist_ok=True)
    pd.to_pickle(result, os.path.join(save_path, f'sentence_embeddings.pkl'))
    

if __name__ == '__main__':
    for model in BERTBasedModels + LLM:
        run(model)