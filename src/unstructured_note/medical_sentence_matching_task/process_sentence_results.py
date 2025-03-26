"""
src/unstructured_note/medical_sentence_matching_task/process_sentence_results.py
Script to process sentence embeddings and compute performance for medical sentence similarity task
"""

import os
from pathlib import Path
import pandas as pd
import numpy as np
from scipy.stats import spearmanr, kendalltau

from unstructured_note.utils.config import MODELS_CONFIG

# Create model lists using config
BERTBasedModels = [model["model_name"] for model in MODELS_CONFIG if model["model_type"] == "BERT"]
LLM = [model["model_name"] for model in MODELS_CONFIG if model["model_type"] == "GPT"]
model_names = BERTBasedModels + LLM

def cosine_similarity(vec1, vec2):
    """Calculate cosine similarity between two vectors"""
    dot_product = np.dot(vec1, vec2)
    norm_vec1 = np.linalg.norm(vec1)
    norm_vec2 = np.linalg.norm(vec2)
    similarity = dot_product / (norm_vec1 * norm_vec2)
    return similarity

def pearson_distance(y_true, y_pred):
    """Calculate Pearson Distance as sqrt(1 - r), where r is Pearson correlation"""
    pearson_corr = np.corrcoef(y_true, y_pred)[0, 1]
    return round(np.sqrt(1 - pearson_corr), 2)

def process_model_results(model_name):
    """Process results for a single model"""
    embedding_path = f"logs/biosses/{model_name}/sentence_embeddings.pkl"
    
    if not os.path.exists(embedding_path):
        print(f"Warning: No embeddings found for {model_name} at {embedding_path}")
        return None
    
    embedding_data = pd.read_pickle(embedding_path)
    
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
    
    # For L1 and L2, convert distances to similarities by negating
    # (smaller distance = higher similarity)
    l1_similarity = -l1_pred
    l2_similarity = -l2_pred
    
    # Calculate correlations for different methods
    results = []
    
    # Cosine similarity results
    results.append({
        'model': model_name,
        'method': 'cosine',
        'pearson_dist': pearson_distance(y_true, cos_pred),
        'pearson_corr': round(np.corrcoef(y_true, cos_pred)[0, 1], 2),
        'spearman_corr': round(spearmanr(y_true, cos_pred)[0], 2),
        'kendall_corr': round(kendalltau(y_true, cos_pred)[0], 2)
    })
    
    # L1 distance results (converted to similarity measure)
    results.append({
        'model': model_name,
        'method': 'l1',
        'pearson_dist': pearson_distance(y_true, l1_similarity),
        'pearson_corr': round(np.corrcoef(y_true, l1_similarity)[0, 1], 2),
        'spearman_corr': round(spearmanr(y_true, l1_similarity)[0], 2),
        'kendall_corr': round(kendalltau(y_true, l1_similarity)[0], 2)
    })
    
    # L2 distance results (converted to similarity measure)
    results.append({
        'model': model_name,
        'method': 'l2',
        'pearson_dist': pearson_distance(y_true, l2_similarity),
        'pearson_corr': round(np.corrcoef(y_true, l2_similarity)[0, 1], 2),
        'spearman_corr': round(spearmanr(y_true, l2_similarity)[0], 2),
        'kendall_corr': round(kendalltau(y_true, l2_similarity)[0], 2)
    })
    
    return results

def process_all_models():
    """Process results for all models and generate performance table"""
    performance_results = []
    
    for model_name in model_names:
        model_results = process_model_results(model_name)
        if model_results:
            pd.DataFrame(model_results).to_csv(f"logs/biosses/{model_name}/results.csv", index=False)
            print(f"Saved performance results for {model_name}")
            performance_results.extend(model_results)
    
    if not performance_results:
        print("No results found to process")
        return
    
    # Create DataFrame and save as CSV
    df = pd.DataFrame(performance_results)
    
    # Create output directory if it doesn't exist
    output_dir = Path("logs/biosses")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_path = output_dir / "all_sentence_performance.csv"
    df.to_csv(output_path, index=False)
    print(f"Saved performance results to {output_path}")

if __name__ == "__main__":
    process_all_models()