import os
from pathlib import Path
from typing import Dict

import pandas as pd
import torch

from utils import get_all_metrics, get_regression_metrics
from config.config import configs

def export_performance(
    config: Dict
):
    src_root = 'logits'
    dst_root = 'performance'

    time = config.get('time', 0)
    if time == 0:
        time_des = 'upon-discharge'
    elif time == 1:
        time_des = '1month'
    elif time == 2:
        time_des = '6months'
    sub_path = os.path.join(config['dataset'], config['task'], config['model'])
    file_name = f'{config["form"]}_{str(config["n_shot"])}shot_{time_des}'
    if config['unit'] is True:
        file_name += '_unit'
    if config['reference_range'] is True:
        file_name += '_range'
    if config.get('prompt_engineering') is True:
        file_name += '_cot'
    impute = config.get('impute')
    if impute == 0:
        file_name += '_no_impute'
    elif impute == 1:
        file_name += '_impute'
    elif impute == 2:
        file_name += '_impute_info'
    src_path = os.path.join(src_root, sub_path, f'{file_name}.pkl')
    dst_path = os.path.join(dst_root, sub_path)
    Path(dst_path).mkdir(parents=True, exist_ok=True)
    
    logits = pd.read_pickle(src_path)

    if config['task'] == 'multitask':   # outcome and readmission tasks
        _labels = logits['labels']
        _preds = logits['preds']
        labels = []
        preds = []
        for label, pred in zip(_labels, _preds):
            if pred[0] != 0.501:
                labels.append(label)
                preds.append(pred)
        _labels, _preds, labels, preds = torch.tensor(_labels), torch.tensor(_preds), torch.tensor(labels), torch.tensor(preds)
        outcome_metrics = get_all_metrics(preds[:, 0], labels[:, 0], 'outcome', None)
        readmission_metrics = get_all_metrics(preds[:, 1], labels[:, 1], 'outcome', None)
        _outcome_metrics = get_all_metrics(_preds[:, 0], _labels[:, 0], 'outcome', None)
        _readmission_metrics = get_all_metrics(_preds[:, 1], _labels[:, 1], 'outcome', None)
        data = {'count': [len(_labels), len(labels)] * 2}
        data = dict(data, **{k: [v1, v2, v3, v4] for k, v1, v2, v3, v4 in zip(_outcome_metrics.keys(), _outcome_metrics.values(), outcome_metrics.values(), _readmission_metrics.values(), readmission_metrics.values())})
        performance = pd.DataFrame(data=data, index=['o all', 'o without unknown samples', 'r all', 'r without unknown samples'])
    elif config['task'] == 'los':
        _labels = logits['labels']
        _preds = logits['preds']
        labels = []
        preds = []
        for label, pred in zip(_labels, _preds):
            if pred[0] != 0:
                labels.append(label)
                preds.append(pred)
        data = {'count': [len(_labels), len(labels)]}
        _labels = torch.vstack([torch.tensor(label).unsqueeze(1) for label in _labels]).squeeze(-1)
        _preds = torch.vstack([torch.tensor(pred).unsqueeze(1) for pred in _preds]).squeeze(-1)
        labels = torch.vstack([torch.tensor(label).unsqueeze(1) for label in labels]).squeeze(-1)
        preds = torch.vstack([torch.tensor(pred).unsqueeze(1) for pred in preds]).squeeze(-1)
        _metrics = get_regression_metrics(_preds, _labels)
        metrics = get_regression_metrics(preds, labels)
        data = dict(data, **{k: [f'{v1:.2f}', f'{v2:.2f}'] for k, v1, v2 in zip(_metrics.keys(), _metrics.values(), metrics.values())})
        performance = pd.DataFrame(data=data, index=['all', 'w/o'])
    else:
        _labels = logits['labels']
        _preds = logits['preds']
        _metrics = get_all_metrics(_preds, _labels, 'outcome', None)
        labels = []
        preds = []
        for label, pred in zip(_labels, _preds):
            if pred != 0.501:
                labels.append(label)
                preds.append(pred)
        metrics = get_all_metrics(preds, labels, 'outcome', None)
        data = {'count': [len(_labels), len(labels)]}
        data = dict(data, **{k: [f'{v1 * 100:.2f}', f'{v2 * 100:.2f}'] for k, v1, v2 in zip(_metrics.keys(), _metrics.values(), metrics.values())})
    
        performance = pd.DataFrame(data=data, index=['all', 'without unknown samples'])
    
    performance.to_csv(os.path.join(dst_path, f'{file_name}.csv'))

if __name__ == '__main__':
    for config in configs:
        export_performance(config)