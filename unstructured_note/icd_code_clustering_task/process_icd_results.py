import os
import json
import torch
import pandas as pd
import numpy as np

from utils.config import LLM, BERTBasedModels


save_dir = 'logs/icd'
file_name = 'icd_embeddings.pkl'
model_names = BERTBasedModels + LLM
ks = [2, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50]


def process():
    performance_table = {'model':[], 'k': [], 'ch': [], 'average_distance': []}
    for model in model_names:
        for k in ks:
            result = pd.read_pickle(os.path.join(save_dir, model, f'icd_result_clusters{k}.pkl'))
            performance_table['model'].append(model)
            performance_table['k'].append(k)
            performance_table['ch'].append(result['ch'])
            ave_dis = sum(result['distance']) / len(result['distance'])
            performance_table['average_distance'].append(ave_dis)
    df = pd.DataFrame(performance_table)
    df.to_csv(os.path.join(save_dir, 'icd_performance.csv'), index=False)