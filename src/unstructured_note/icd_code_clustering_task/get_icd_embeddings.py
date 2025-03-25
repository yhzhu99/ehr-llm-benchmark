import os

from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from tqdm import tqdm
from accelerate import Accelerator
import torch

from utils.config import BERTBasedModels, LLM, LLMPathList

class MyDataset(Dataset):
    def __init__(self, dataset_path='datasets/icd10cm_order_2023.txt'):
        # 指定分割索引
        split_index = [6, 14, 16, 77]
        # 初始化一个空列表用于存储数据
        data = []
        # 打开文件读取数据
        with open(dataset_path, 'r') as f:
            content = f.readline()
            while content:
                # 按照索引分割内容并去除空白
                contents = [content[split_index[0]:split_index[1]].strip(),
                            content[split_index[1]:split_index[2]].strip(),
                            content[split_index[2]:split_index[3]].strip(),
                            content[split_index[3]:].strip()]
                code = contents[0]
                disease = contents[3]
                # 处理code长度小于等于4的情况
                if len(code) <= 4:
                    # print(code)
                    # 将处理后的数据存入列表
                    data.append([code, disease])
                content = f.readline()

        self.data = data

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx][0], self.data[idx][1] # code and disease name

# accelerator = Accelerator()
# device = accelerator.device
# device = torch.device('cuda:1') if torch.cuda.is_available() else torch.device("cpu") 

def run(model_name):
    dataset = MyDataset()
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
    save_dir = 'logs/icd'
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
    for batch in tqdm(dataloader, desc=f'Running ICD task for model:{model_name}'):
        code, disease = batch
        encoded_input1 = tokenizer(disease, padding="max_length", max_length=512, truncation=True, return_tensors="pt").to(device)
        if model_name in LLM:
            outputs = model(**encoded_input1, output_hidden_states=True)
            embedding = outputs.hidden_states[-1][0, -1, :].detach().cpu()
        else:
            outputs = model(**encoded_input1)
            embedding = outputs.last_hidden_state[0, 0, :].detach().cpu()
        result.append({
            'code': code,
            'disease': disease,
            'embedding': embedding,
        })
    save_path = os.path.join(save_dir, model_name)
    os.makedirs(save_path, exist_ok=True)
    pd.to_pickle(result, os.path.join(save_path, f'icd_embeddings.pkl'))


if __name__ == '__main__':
    for model in BERTBasedModels + LLM:
        run(model)
