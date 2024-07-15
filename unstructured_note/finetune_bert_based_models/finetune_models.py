import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import csv
import json
import random
import argparse
import jsonlines
import pandas as pd
import numpy as np
import pathlib
import copy

from tqdm import tqdm

from sklearn.utils import shuffle
from sklearn.metrics import roc_curve, auc, roc_auc_score, accuracy_score, \
    precision_score, recall_score, f1_score, precision_recall_curve

from transformers import AutoTokenizer, GPT2Tokenizer, GPT2Config
from transformers import AutoModelForSequenceClassification, GPT2ForSequenceClassification
from transformers import get_scheduler, set_seed

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from torchmetrics import AveragePrecision
from torch.cuda.amp import GradScaler, autocast
# from optimum.bettertransformer import BetterTransformer
# from optimum.int8 import load_int8_model, prepare_int8_model
import bitsandbytes as bnb

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'


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

def evaluateResultClass(y_pred, y_test):
    acc = accuracy_score(y_test, [round(y) for y in y_pred])
    pre = precision_score(y_test, [round(y) for y in y_pred])
    rec = recall_score(y_test, [round(y) for y in y_pred])
    f1 = f1_score(y_test, [round(y) for y in y_pred])
    auroc = roc_auc_score(y_test, y_pred)
    auprc_fun = AveragePrecision(task="binary")
    auprc_fun(torch.from_numpy(y_pred), torch.from_numpy(y_test))
    auprc = auprc_fun.compute().item()
    minpse_score = minpse(y_pred, y_test)
    return acc, pre, rec, f1, auprc, auroc, minpse_score

def minpse(preds, labels):
    precisions, recalls, thresholds = precision_recall_curve(labels, preds)
    minpse_score = np.max([min(x, y) for (x, y) in zip(precisions, recalls)])
    return minpse_score


parser = argparse.ArgumentParser(description='Demo of argparse')
parser.add_argument('--model', type=str, default='BERT', choices=['BERT', 'ClinicalBERT', 'ClinicalBERT', 'BioBERT', 'GatorTron', 'Clinical-Longformer', 'GPT-2', 'BioGPT', 'MedAlpaca', 'HuatuoGPT', 'OpenBioLLM', 'meditron'])
parser.add_argument('--dataset', type=str, default='discharge', choices=['discharge', 'noteevent'])
parser.add_argument('--cuda', type=str, default='0', choices=['0', '1'])
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--freeze', type=bool, default=False)
parser.add_argument('--embedding', type=bool, default=False)
args = parser.parse_args()

num_epochs = 50
learning_rate = 1e-5
early_stop_cnt = 0
max_stop_epochs = 5
accumulation_steps = 200

no_token_type_id_list = ['ClinicalBERT', 'BioGPT', 'MedAlpaca', 'HuatuoGPT', 'OpenBioLLM', 'meditron']
# no_to_device_list = ['MedAlpaca', 'HuatuoGPT']

if args.model == 'BERT':
    tokenizer = AutoTokenizer.from_pretrained("./HF_models/BERT")
    model = AutoModelForSequenceClassification.from_pretrained("./HF_models/BERT", num_labels=2, output_hidden_states=True)
    batch_size = args.batch_size
    
elif args.model == 'ClinicalBERT':
    tokenizer = AutoTokenizer.from_pretrained("./HF_models/ClinicalBERT")
    model = AutoModelForSequenceClassification.from_pretrained("./HF_models/ClinicalBERT", num_labels=2, output_hidden_states=True)
    batch_size = args.batch_size
    
elif args.model == 'BioBERT':
    tokenizer = AutoTokenizer.from_pretrained("./HF_models/BioBERT")
    model = AutoModelForSequenceClassification.from_pretrained("./HF_models/BioBERT", num_labels=2, output_hidden_states=True)
    batch_size = args.batch_size
    
elif args.model == 'GatorTron':
    tokenizer = AutoTokenizer.from_pretrained("./HF_models/GatorTron")
    model = AutoModelForSequenceClassification.from_pretrained("./HF_models/GatorTron", num_labels=2, output_hidden_states=True)
    batch_size = args.batch_size
    
elif args.model == 'Clinical-Longformer':
    tokenizer = AutoTokenizer.from_pretrained("./HF_models/Clinical-Longformer")
    model = AutoModelForSequenceClassification.from_pretrained("./HF_models/Clinical-Longformer", num_labels=2, output_hidden_states=True)
    batch_size = args.batch_size
    
elif args.model == 'GPT-2':
    # model_config = GPT2Config.from_pretrained(pretrained_model_name_or_path="./HF_models/GPT-2/config.json", num_labels=2)
    # tokenizer = GPT2Tokenizer.from_pretrained("./HF_models/GPT-2")
    # model = GPT2ForSequenceClassification.from_pretrained("./HF_models/GPT-2", config=model_config)
    # tokenizer.pad_token = tokenizer.eos_token
    tokenizer = AutoTokenizer.from_pretrained("./HF_models/GPT-2")
    model = AutoModelForSequenceClassification.from_pretrained("./HF_models/GPT-2", num_labels=2, output_hidden_states=True)
    tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = tokenizer.eos_token_id  # 确保模型配置中有 pad_token_id，否则bs>1时报错
    batch_size = args.batch_size
    
elif args.model == 'BioGPT':
    tokenizer = AutoTokenizer.from_pretrained("./HF_models/BioGPT")
    model = AutoModelForSequenceClassification.from_pretrained("./HF_models/BioGPT", num_labels=2, output_hidden_states=True)
    tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = tokenizer.eos_token_id
    batch_size = args.batch_size
    
elif args.model == 'MedAlpaca':
    tokenizer = AutoTokenizer.from_pretrained("./HF_models/MedAlpaca")
    model = AutoModelForSequenceClassification.from_pretrained("./HF_models/MedAlpaca", num_labels=2, output_hidden_states=True, load_in_8bit=True)
    tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = tokenizer.eos_token_id
    # model = bnb.nn.quantization.Int8Params(model)
    batch_size = args.batch_size
    # print(learning_rate)
    
elif args.model == 'HuatuoGPT':
    from transformers import AutoConfig, AutoModelForCausalLM
    model_config = AutoConfig.from_pretrained("./HF_models/HuatuoGPT", trust_remote_code=True, num_labels=2)
    tokenizer = AutoTokenizer.from_pretrained("./HF_models/HuatuoGPT", trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained("./HF_models/HuatuoGPT", config=model_config, trust_remote_code=True, load_in_8bit=True)
    tokenizer.pad_token = tokenizer.eos_token  # 设置 pad_token 为 eos_token
    model.config.pad_token_id = tokenizer.eos_token_id
    batch_size = args.batch_size
    
elif args.model == 'OpenBioLLM':
    tokenizer = AutoTokenizer.from_pretrained("./HF_models/OpenBioLLM")
    model = AutoModelForSequenceClassification.from_pretrained("./HF_models/OpenBioLLM", num_labels=2, output_hidden_states=True)
    tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = tokenizer.eos_token_id
    batch_size = args.batch_size
    
elif args.model == 'meditron':
    tokenizer = AutoTokenizer.from_pretrained("./HF_models/meditron")
    model = AutoModelForSequenceClassification.from_pretrained("./HF_models/meditron", num_labels=2, output_hidden_states=True)
    tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = tokenizer.eos_token_id
    batch_size = args.batch_size

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
            # 遍历csv_reader对象的每一行内容并输出
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

max_token_length = 512
train_data_content = pd.DataFrame({"content": Data['train']['text'], "label": Data['train']['label']})
train_data_content = shuffle(train_data_content, random_state=42)
train_data = tokenizer(train_data_content.content.to_list(), padding="max_length", max_length=max_token_length, truncation=True,
                       return_tensors="pt")
train_label = train_data_content.label.to_list()
train_dataset = TensorDataset(train_data["input_ids"], train_data["attention_mask"], torch.tensor(train_label))
train_sampler = RandomSampler(train_dataset)
train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=batch_size)

valid_data_content = pd.DataFrame({"content": Data['valid']['text'], "label": Data['valid']['label']})
# valid_data_content = shuffle(valid_data_content, random_state=42)
valid_data = tokenizer(valid_data_content.content.to_list(), padding="max_length", max_length=max_token_length, truncation=True,
                       return_tensors="pt")
valid_label = valid_data_content.label.to_list()
valid_dataset = TensorDataset(valid_data["input_ids"], valid_data["attention_mask"], torch.tensor(valid_label))
valid_sampler = SequentialSampler(valid_dataset)
valid_dataloader = DataLoader(valid_dataset, sampler=valid_sampler, batch_size=batch_size)

test_data_content = pd.DataFrame({"content": Data['test']['text'], "label": Data['test']['label']})
# test_data_content = shuffle(test_data_content, random_state=42)
test_data = tokenizer(test_data_content.content.to_list(), padding="max_length", max_length=max_token_length, truncation=True,
                      return_tensors="pt")
test_label = test_data_content.label.to_list()
test_dataset = TensorDataset(test_data["input_ids"], test_data["attention_mask"], torch.tensor(test_label))
test_sampler = SequentialSampler(test_dataset)
test_dataloader = DataLoader(test_dataset, sampler=test_sampler, batch_size=batch_size)

# optimizer = bnb.optim.Adam8bit(model.parameters(), lr=learning_rate)
optimizer = AdamW(model.parameters(), lr=learning_rate)
num_training_steps = num_epochs * len(train_dataloader)
lr_scheduler = get_scheduler(
    name="linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps
)

if args.freeze:
    for param in model.base_model.parameters():
        param.requires_grad = False

# 使用混合精度
# scaler = GradScaler()

device = torch.device(f"cuda:{args.cuda}") if torch.cuda.is_available() else torch.device("cpu") 
# if args.model not in no_to_device_list:  
model.to(device)
result = {
    'loss': [],
    'loss_mini': [],
    'best_epoch': 0,
    'best_valid_auroc': 0,
    # TODO: add param minpse
    'valid_result': {'acc': [], 'pre': [], 'rec': [], 'f1': [], 'auprc': [], 'auroc': [], 'minpse': []},
    'test_result': {'acc': None, 'pre': None, 'rec': None, 'f1': None, 'auprc': None, 'auroc': None, 'minpse': None, 'y_pred': None, 'y_label': None}
}

# help(model)
for epoch in range(num_epochs):
    total_loss = 0
    model.train()
    print('#' * 20, 'train model', '#' * 20)
    for step, batch in tqdm(enumerate(train_dataloader)):
        b_input_ids = batch[0].to(device)
        b_input_mask = batch[1].to(device)
        b_labels = batch[2].to(device)
        # print(b_labels)
        model.zero_grad()
        # model.zero_grad() if step % accumulation_steps == 0 else None
        if args.model in no_token_type_id_list:
            outputs = model(b_input_ids,
                        labels=b_labels,
                        attention_mask=b_input_mask, )
        else:
            outputs = model(b_input_ids,
                            token_type_ids=None,
                            labels =b_labels,
                            attention_mask=b_input_mask, )
        loss = outputs.loss
        # print(loss)
        if loss is None:
            print(f"Step {step}: Loss is None. Skipping this step.")
            continue
        total_loss += loss.item()
        result['loss_mini'].append(loss.item())
        loss.backward()
        # torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        lr_scheduler.step()
        torch.cuda.empty_cache()
        
    avg_train_loss = total_loss / len(train_dataloader)
    result['loss'].append(avg_train_loss)
    print("epoch: {}, avg_loss: {}".format(epoch + 1, avg_train_loss))

    model.eval()

    with torch.no_grad():
        y_preds = []
        y_labels = []       
        print('#' * 20, 'eval model', '#' * 20)
        for step, batch in tqdm(enumerate(valid_dataloader)):
            b_input_ids = batch[0].to(device)
            b_input_mask = batch[1].to(device)
            b_labels = batch[2].to(device)
            if args.model in no_token_type_id_list:
                outputs = model(b_input_ids,
                            labels=b_labels,
                            attention_mask=b_input_mask, )
            else:
                outputs = model(b_input_ids,
                                token_type_ids=None,
                                labels =b_labels,
                                attention_mask=b_input_mask, )
                
            y_pred = nn.Softmax(dim=1)(outputs["logits"]).cpu().numpy()[:, 1]
            y_label = b_labels.cpu().numpy()
            y_preds.append(y_pred)
            y_labels.append(y_label)
        y_preds = np.concatenate(y_preds)
        y_labels = np.concatenate(y_labels)
        acc, pre, rec, f1, auprc, auroc, minpse_score = evaluateResultClass(np.squeeze(y_preds), np.squeeze(y_labels))
        result['valid_result']['acc'].append(acc)
        result['valid_result']['pre'].append(pre)
        result['valid_result']['rec'].append(rec)
        result['valid_result']['f1'].append(f1)
        result['valid_result']['auprc'].append(auprc)
        result['valid_result']['auroc'].append(auroc)
        result['valid_result']['minpse'].append(minpse_score)

        if auroc > result['best_valid_auroc']:
            
            early_stop_cnt = 0
            result['best_valid_auroc'] = auroc
            result['best_epoch'] = epoch

            print('best Valid result, epoch: {}, acc: {}, pre: {}, rec: {}, f1: {}, auprc: {}, auroc: {}, minpse: {}'.format(
                epoch, acc, pre, rec, f1, auprc, auroc, minpse_score
            ))
            
            if args.freeze:
                model_path = './logs/freeze/{}/{}-{}-epoch-best.pt'.format(args.model, args.model, args.dataset)
            else:
                model_path = './logs/finetune/{}/{}-{}-epoch-best.pt'.format(args.model, args.model, args.dataset)
            pathlib.Path(os.path.dirname(model_path)).mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), model_path)
            
            best_result = copy.deepcopy(result)
            best_epoch = epoch
            y_pred = []
            y_label = []
            print('#' * 20, 'test model', '#' * 20)
        else:
            early_stop_cnt = early_stop_cnt + 1
    if early_stop_cnt == max_stop_epochs:
        break
     
model.load_state_dict(torch.load(model_path))

# if args.model not in no_to_device_list:  
model.to(device)       
model.eval()

embeddings = []
y_preds = []
y_labels = []
with torch.no_grad():
    for step, batch in tqdm(enumerate(test_dataloader)):
        b_input_ids = batch[0].to(device)
        b_input_mask = batch[1].to(device)
        b_labels = batch[2].to(device)
        if args.model in no_token_type_id_list:
            outputs = model(b_input_ids,
                        labels=b_labels,
                        attention_mask=b_input_mask, )
        else:
            outputs = model(b_input_ids,
                            token_type_ids=None,
                            labels =b_labels,
                            attention_mask=b_input_mask, )

        # if len(y_pred) == 0:
        #     y_pred = nn.Softmax(dim=1)(outputs["logits"]).cpu().numpy()[:, 1]
        #     y_label = b_labels.cpu().numpy()
        # else:
        #     y_pred = np.vstack((y_pred, nn.Softmax(dim=1)(outputs["logits"]).cpu().numpy()[:, 1]))
        #     y_label = np.vstack((y_label, b_labels.cpu().numpy()))
        
        y_pred = nn.Softmax(dim=1)(outputs["logits"]).cpu().numpy()[:, 1]
        y_label = b_labels.cpu().numpy()        
        y_preds.append(y_pred)
        y_labels.append(y_label)
        
        # TODO: concat embeddings
        if args.embedding:
            batch_embedding = outputs.hidden_states[-1][:, 0, :]
            # print(batch_embedding)
            batch_embedding = batch_embedding.detach().cpu()
            del b_input_ids, b_input_mask, outputs
            embeddings.append(batch_embedding)
    
    # y_preds = [torch.from_numpy(arr) for arr in y_preds]
    # y_preds = torch.cat(y_preds,dim=0) 
    # y_labels = [torch.from_numpy(arr) for arr in y_labels]  
    # y_labels = torch.cat(y_labels, dim = 0) 
    y_preds = np.concatenate(y_preds)
    y_labels = np.concatenate(y_labels) 
    acc, pre, rec, f1, auprc, auroc, minpse_score = evaluateResultClass(np.squeeze(y_preds), np.squeeze(y_labels))
    best_result['test_result']['acc'] = acc
    best_result['test_result']['pre'] = pre
    best_result['test_result']['rec'] = rec
    best_result['test_result']['f1'] = f1
    best_result['test_result']['auprc'] = auprc
    best_result['test_result']['auroc'] = auroc
    best_result['test_result']['minpse'] = minpse_score
    best_result['test_result']['y_pred'] = y_preds.tolist()
    best_result['test_result']['y_label'] = y_labels.tolist()
    
    print('Test result, epoch: {}, acc: {}, pre: {}, rec: {}, f1: {}, auprc: {}, auroc: {}, minpse: {}'.format(
        best_epoch, acc, pre, rec, f1, auprc, auroc, minpse_score
    ))
    
# embeddings = torch.cat(embeddings,dim=0)

if args.freeze:
    # embedding_path = './logs/freeze/{}/note_embedding_{}_{}_all_cls.pkl'.format(args.model, args.dataset, best_epoch)
    result_path = './logs/freeze/{}/note_{}_{}_{}_results.pkl'.format(args.model, args.model, args.dataset, best_epoch)
else:
    # embedding_path = './logs/finetune/{}/note_embedding_{}_{}_all_cls.pkl'.format(args.model, args.dataset, best_epoch)
    result_path = './logs/finetune/{}/note_{}_{}_{}_results.pkl'.format(args.model, args.model, args.dataset, best_epoch)


# pd.to_pickle(embeddings, embedding_path)
pd.to_pickle(best_result, result_path)

# print("embeddings:", embeddings)

if args.embedding:
    embeddings = torch.cat(embeddings,dim=0)
    embeddings = embeddings.numpy()
    
    if args.freeze:
        embedding_path = './logs/freeze/{}/note_embedding_{}_{}_all_cls.pkl'.format(args.model, args.dataset, best_epoch)
    else:
        embedding_path = './logs/finetune/{}/note_embedding_{}_{}_all_cls.pkl'.format(args.model, args.dataset, best_epoch)
    pd.to_pickle(embeddings, embedding_path)
