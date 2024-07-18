import re
import os

import pandas as pd
import torch

from utils.binary_classification_metrics import get_binary_metrics
from utils.config import LLM

save_dir = 'logs/generation'
datasets = ['noteevent', 'discharge']


def extract_number(s):
    # 匹配0-1之间的小数或百分数
    decimal_pattern = r'\b0(?:\.\d+)?\b|\b1(?:\.0+)?\b'
    percent_pattern = r'\b0?\.\d+%|\b100%|\b\d{1,2}%'
    
    # 查找所有符合条件的匹配项
    decimals = re.findall(decimal_pattern, s)
    percents = re.findall(percent_pattern, s)
    
    # 检查是否有多个小数
    if len(decimals) > 1:
        raise ValueError("Multiple decimals found in the string.")
    
    # 提取并转换
    if decimals:
        return float(decimals[0])
    elif percents:
        percent_value = percents[0]
        return float(percent_value.strip('%')) / 100.0
    
    # 如果没有匹配项
    raise ValueError("No decimal or percent found in the string.")


def process():
    performance = {'model': [], 'dataset': [], 'auroc': [], 'auprc': [], 'minpse': [], 'missingCount': [], 'missingRate': []}
    for model in LLM:
        for dataset in datasets:
            result = pd.read_pickle(os.path.join(save_dir, model, dataset, 'output_test.pkl'))
            preds = []
            labels = []
            missing = 0
            for i, item in enumerate(result):
                try:
                    preds.append(extract_number(item['output']))
                except Exception as e: 
                    preds.append(0.5)
                    missing += 1
                labels.append(float(item['label']))
            pd.to_pickle({'preds': preds, 'labels': labels}, os.path.join(save_dir, model, dataset, 'output_logits.pkl'))
            preds = torch.tensor(preds)
            labels = torch.tensor(labels)
            perf = get_binary_metrics(preds, labels)
            performance['model'].append(model)
            performance['dataset'].append(dataset)
            performance['auroc'].append(perf['auroc'])
            performance['auprc'].append(perf['auprc'])
            performance['minpse'].append(perf['minpse'])
            performance['missingRate'].append(missing/len(result))
            performance['missingCount'].append(missing)
            # metrics = run_bootstrap(preds, labels)
            # for k, v in metrics.items():
            #     if k in ['auroc', 'auprc']:
            #         mean_var = 100 * v['mean']
            #         std_var = 100*v['std']
            #         print(f"{mean_var:.2f}±{std_var:.2f}", end=' & ')
    df = pd.DataFrame(performance)
    df[['auroc', 'auprc', 'minpse', 'missingRate']] = df[['auroc', 'auprc', 'minpse', 'missingRate']].apply(lambda x: round(x * 100, 2))
    df.to_csv(f'{save_dir}/performance.csv', index=False)

process()