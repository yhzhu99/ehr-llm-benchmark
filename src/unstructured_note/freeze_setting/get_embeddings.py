from pathlib import Path
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
import argparse
import random

from tqdm import tqdm
import pandas as pd
import numpy as np
import torch
import jsonlines
import csv
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModel, set_seed
from accelerate import Accelerator

from utils.config import BERTBasedModels, LLM, LLMPathList

torch.cuda.empty_cache()

def seed_torch(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False


seed_torch(42)
set_seed(42)

MAX_LENGTH = 512

parser = argparse.ArgumentParser(description='Demo of argparse')
parser.add_argument('--model', type=str, default='bert', choices=BERTBasedModels + LLM)
parser.add_argument('--dataset', type=str, default='discharge', choices=['discharge', 'noteevent'])
parser.add_argument('--cuda', type=int, default=0, choices=[0,1,2,3,4,5,6,7])
parser.add_argument('--batch_size', type=int, default=1)
args = parser.parse_args()
model_name = args.model

if model_name in LLM:
    # Initialize accelerator
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
elif model_name in BERTBasedModels:
    device = torch.device('cuda:1') if torch.cuda.is_available() else torch.device("cpu")

    model_path = f'HF_models/{model_name}'
    model = AutoModel.from_pretrained(model_path, trust_remote_code=True).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
else:
    raise ValueError(f"Model {model_name} not supported.")

# 数据处理
Data = {'train': {}, 'valid': {}, 'test': {}}
for key in Data.keys():
    Data[key] = {'ID': [], 'text': [], 'label': []}
if args.dataset == 'discharge':
    file_names = ['train', 'val', 'test']
    file_dir = r'./my_datasets/discharge'
    save_dir = r'./my_datasets/discharge'

    for file_name in file_names:
        file_path = os.path.join(file_dir, '{}.csv'.format(file_name))
        with open(file_path, "r") as csvfile:
            csv_reader = csv.reader(csvfile, delimiter=',')
            header = next(csv_reader)
            for row in csv_reader:
                key = 'valid' if file_name == 'val' else file_name
                Data[key]['ID'].append(int(float(row[1])))
                Data[key]['text'].append(row[2])
                assert int(float(row[3])) == 0 or int(float(row[3])) == 1
                Data[key]['label'].append(int(float(row[3])))
elif args.dataset == 'noteevent':
    file_names = ['train', 'valid', 'test']
    file_dir = r'./datasets/noteevent'
    save_dir = r'./datasets/noteevent'
    error_dict = {}
    for file_name in file_names:
        file_path = os.path.join(file_dir, '{}-text.json'.format(file_name))
        with open(file_path) as file:
            for disease_info in jsonlines.Reader(file):
                key = file_name
                Data[key]['ID'].append(disease_info["id"])
                assert isinstance(disease_info["texts"], str) or isinstance(disease_info["texts"], list)
                if isinstance(disease_info["texts"], list):
                    long_text = ''
                    for short_text in disease_info["texts"]:
                        assert isinstance(short_text, str) or isinstance(short_text, list)
                        if isinstance(short_text, list):
                            long_test_2 = ''
                            for short_text_2 in short_text:
                                assert isinstance(short_text_2, str)
                                long_test_2 += short_text_2
                            long_text += long_test_2
                        else:
                            long_text += short_text
                    Data[key]['text'].append(long_text)
                else:
                    Data[key]['text'].append(disease_info["texts"])
                assert disease_info["label"] == 0 or disease_info["label"] == 1
                Data[key]['label'].append(int(disease_info["label"]))

def save_embedding(mode='train'):
    all_embeddings_train = []
    # Iterate over texts and labels together
    # cnt = 0
    for text, label in tqdm(zip(Data[mode]['text'], Data[mode]['label']), total=len(Data[mode]['text']), desc=f"Processing {mode} data"):
        # cnt+=1
        # if cnt == 10: break
        # Tokenize the input text
        text = text[:1024]
        input_tokens = tokenizer(text,
                                 return_tensors="pt",
                                 return_attention_mask=False,
                                 truncation=True,
                                 max_length=MAX_LENGTH,
                                 padding=True)

        input_ids = input_tokens['input_ids'].to(device)

        with torch.no_grad():
            if model_name in LLM:
                outputs = model(input_ids, output_hidden_states=True)
                last_token_embedding = outputs.hidden_states[-1][0, -1, :].detach().cpu()
            elif model_name in BERTBasedModels:
                outputs = model(input_ids)
                last_token_embedding = outputs.last_hidden_state[0, 0, :].detach().cpu()

        # Create a dictionary with the embedding and the corresponding label
        embedding_dict = {'embedding': last_token_embedding, 'label': label}

        all_embeddings_train.append(embedding_dict)

    embedding_path = f"logs/embeddings/{model_name}/{args.dataset}/embed_{mode}.pkl"
    # 确保目标文件夹存在
    embedding_folder = Path(embedding_path).parent
    if not embedding_folder.exists():
        embedding_folder.mkdir(parents=True, exist_ok=True)
    pd.to_pickle(all_embeddings_train, embedding_path)

save_embedding('train')
save_embedding('valid')
save_embedding('test')
