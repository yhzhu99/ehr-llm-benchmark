from langchain_huggingface import HuggingFacePipeline
from transformers import AutoModelForCausalLM, AutoTokenizer,pipeline
from pathlib import Path
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
import pandas as pd
import numpy as np
import torch
import jsonlines
import random
import argparse
from torch.utils.data import DataLoader, TensorDataset, RandomSampler, SequentialSampler
from sklearn.utils import shuffle
from accelerate import Accelerator
import csv
from tqdm import tqdm
from transformers import get_scheduler, set_seed
import re
from utils.config import LLM, LLMPathList

parser = argparse.ArgumentParser(description='Demo of argparse')
parser.add_argument('--dataset', type=str, default='discharge', choices=['discharge', 'noteevent'])
parser.add_argument('--model', type=str, default='OpenBioLLM')
# parser.add_argument('--cuda', type=int, default=0, choices=[0,1,2,3,4,5,6,7])
args = parser.parse_args()

# Initialize accelerator
accelerator = Accelerator()
device = accelerator.device

discharge_prompt = "Based on the intensive care clinical notes, please predict the patient's readmission probability. 1 for readmission, 0 for no readmission. The closer to 1, the more likely the patient will be readmitted. Please output the probability from 0 to 1. Please directly output the probability number, do not explain anything else."
noteevent_prompt = "Based on the intensive care clinical notes, please predict the patient's mortality outcome. 1 for dead, 0 for alive. The closer to 1, the more likely the patient will die. Please output the probability from 0 to 1. Please directly output the probability number, do not explain anything else."

instruction_prompt = discharge_prompt
if args.dataset=='noteevent':
    instruction_prompt = noteevent_prompt

if args.model in LLM:
    model_path = LLMPathList[args.model]
else:
    raise ValueError(f"Model {args.model} not found in LLM list")


if args.model not in ['BioGPT', 'GPT-2']:
    model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True, device_map="auto")
    model = accelerator.prepare(model)
else:
    device = torch.device("cuda:1")
    model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True).to(device)

if args.model in ["GPT-2"]:
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, device_map="auto", truncation=True, max_length=512)
else:
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, device_map="auto", truncation=True, max_length=4096)
tokenizer.pad_token = tokenizer.eos_token
model.config.pad_token_id = tokenizer.eos_token_id


pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, max_new_tokens=20, top_k=50, temperature=0.1)
llm = HuggingFacePipeline(pipeline=pipe)

Data = {'train': {}, 'valid': {}, 'test': {}}
for key in Data.keys():
    Data[key] = {'ID': [], 'text': [], 'label': []}
if args.dataset == 'discharge':
    file_names = ['train', 'val', 'test']
    file_dir = r'./datasets/discharge'
    save_dir = r'./datasets/discharge'

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


def extract_after_answer(input_string):
    match = re.search(r'Answer is:\s*(.*)', input_string)
    if match:
        return match.group(1).strip()
    return None

def get_logits(mode='test'): # only test set is required
    all_embeddings_train = []
    # Iterate over texts and labels together
    # cnt = 0
    for text, label in tqdm(zip(Data[mode]['text'], Data[mode]['label']), total=len(Data[mode]['text']), desc=f"Processing {mode} data"):
        # cnt+=1
        # if cnt<=1657: continue
        # Create a dictionary with the embedding and the corresponding label
        # text = text[:2048]
        input_prompt = instruction_prompt + "\n" + text + "\nPlease directly output the probability number, do not explain anything else.\nAnswer is: "
#         input_prompt = """
# < |im_start| > system
# You are a medical doctor answering real-world medical entrance exam questions. Based
# on your understanding of basic and clinical science, medical knowledge, and mechanisms
# underlying health, disease, patient care, and modes of therapy, answer the following multiplechoice question. Select one correct answer from A to D. Base your answer on the current and
# standard practices referenced in medical guidelines. < |im_end| >‘
# < |im_start| > question
# Question: Which of the following ultrasound findings has the highest association with
# aneuploidy?
# Options:
# (A) Choroid plexus cyst
# (B) Nuchal translucency
# (C) Cystic hygroma
# (D) Single umbilical artery < |im_end| >
# < |im_start| > answer"""
        result = llm.invoke(input_prompt)
        # print(result, label)
        extracted = extract_after_answer(result)
        # print(extracted, label)
        output_dict = {'output': extracted, 'label': label}
        all_embeddings_train.append(output_dict)

    embedding_path = f"logs/generation/{args.model}/{args.dataset}/output_{mode}.pkl"
    embedding_folder = Path(embedding_path).parent
    if not embedding_folder.exists():
        embedding_folder.mkdir(parents=True, exist_ok=True)
    pd.to_pickle(all_embeddings_train, embedding_path)

get_logits()
