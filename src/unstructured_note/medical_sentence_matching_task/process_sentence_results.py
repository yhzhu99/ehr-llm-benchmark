import os
import pandas as pd
import numpy as np
from scipy.stats import spearmanr, kendalltau

from utils.config import LLM, BERTBasedModels

def cosine_similarity(vec1, vec2):
    dot_product = np.dot(vec1, vec2)
    norm_vec1 = np.linalg.norm(vec1)
    norm_vec2 = np.linalg.norm(vec2)
    similarity = dot_product / (norm_vec1 * norm_vec2)
    return similarity


save_dir = 'logs/sentence'
model_names = BERTBasedModels + LLM


def process():
    performance_table = {'model':[], 'method': [], 'pearson_corr': [], 'spearman_corr': [], 'kendall_corr': []}
    for model_name in model_names:
        embedding_data = pd.read_pickle(os.path.join(save_dir, model_name, 'sentence_embeddings.pkl'))
        y_true = []
        cos_pred = []
        l1_pred = []
        l2_pred = []
        for item in embedding_data:
            embedding1, embedding2, score = item['embedding1'], item['embedding2'], item['score']
            embedding1 = embedding1.numpy()
            embedding2 = embedding2.numpy()
            y_true.append(score)
            cos_pred.append(cosine_similarity(embedding1, embedding2).item())
            l1_pred.append(np.linalg.norm(embedding1 - embedding2, ord=1))
            l2_pred.append(np.linalg.norm(embedding1 - embedding2, ord=2))
        y_true = np.array(y_true)
        cos_pred = np.array(cos_pred)
        l1_pred = np.array(l1_pred)
        l2_pred = np.array(l2_pred)

        performance_table['model'].append(model_name)
        performance_table['model'].append(model_name)
        performance_table['model'].append(model_name)
        performance_table['method'].append('cosine')
        performance_table['pearson_corr'].append(round(np.corrcoef(y_true, cos_pred)[0, 1], 2))
        performance_table['spearman_corr'].append(round(spearmanr(y_true, cos_pred)[0], 2))
        performance_table['kendall_corr'].append(round(kendalltau(y_true, cos_pred)[0], 2))
        performance_table['method'].append('l1')
        performance_table['pearson_corr'].append(round(np.corrcoef(y_true, l1_pred)[0, 1], 2))
        performance_table['spearman_corr'].append(round(spearmanr(y_true, l1_pred)[0], 2))
        performance_table['kendall_corr'].append(round(kendalltau(y_true, l1_pred)[0], 2))
        performance_table['method'].append('l2')
        performance_table['pearson_corr'].append(round(np.corrcoef(y_true, l2_pred)[0, 1], 2))
        performance_table['spearman_corr'].append(round(spearmanr(y_true, l2_pred)[0], 2))
        performance_table['kendall_corr'].append(round(kendalltau(y_true, l2_pred)[0], 2))
    df = pd.DataFrame(performance_table)
    df.to_csv(os.path.join(save_dir, 'sentence_performance.csv'), index=False)