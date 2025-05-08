"""
src/unstructured_note/icd_code_clustering_task/process_icd_results.py
Script to perform clustering on ICD embeddings and process results
"""

import numpy as np
import pandas as pd
import time
import argparse
from pathlib import Path
import os

from sklearn.cluster import KMeans
from sklearn.metrics import calinski_harabasz_score

from unstructured_note.utils.config import MODELS_CONFIG

# Create model lists
BERT_MODELS = [model["model_name"] for model in MODELS_CONFIG if model["model_type"] == "BERT"]
LLM_MODELS = [model["model_name"] for model in MODELS_CONFIG if model["model_type"] == "GPT"]
EMBEDDING_MODELS = [model["model_name"] for model in MODELS_CONFIG if model["model_type"] == "embedding"]

parser = argparse.ArgumentParser(description='Process ICD embeddings with clustering')
parser.add_argument('--model', type=str, default='BERT', choices=BERT_MODELS + LLM_MODELS + EMBEDDING_MODELS)
parser.add_argument('--process_all', action='store_true', help='Process all models and generate performance table')
parser.add_argument('--ks', type=str, default='10,20,30,40,50', help='Comma-separated list of cluster numbers')
args = parser.parse_args()

def calculate_icd_distance(code1, code2):
    """
    Calculate the distance between two ICD codes in the ICD tree hierarchy.

    Distance is calculated as the sum of:
    1. The distance from code1 to the lowest common ancestor (LCA)
    2. The distance from code2 to the LCA
    """
    # Find the common prefix (LCA)
    min_len = min(len(code1), len(code2))
    lca_len = 0

    for i in range(min_len):
        if code1[i] == code2[i]:
            lca_len += 1
        else:
            break

    # Calculate distance from each code to the LCA
    dist1 = len(code1) - lca_len
    dist2 = len(code2) - lca_len

    # Total distance
    return dist1 + dist2

def run_clustering(model_name, k_values):
    """Run clustering with specified k values for a given model"""
    print(f"Running clustering for model: {model_name}")

    # Load embeddings
    embedding_path = f'logs/icd/{model_name}/icd_embeddings.pkl'
    if not os.path.exists(embedding_path):
        print(f"Embedding file not found: {embedding_path}")
        return

    embeddings = pd.read_pickle(embedding_path)

    # Extract codes and embeddings
    embedding_list = []
    icd_code_list = []
    for disease_info in embeddings:
        code = disease_info["code"]
        embedding = disease_info["embedding"]
        icd_code_list.append(code)
        embedding_list.append(embedding)

    # Ensure output directory exists
    output_dir = Path(f'logs/icd/{model_name}')
    output_dir.mkdir(parents=True, exist_ok=True)

    # Process each k value
    for k in k_values:
        print(f'Processing k = {k} for model {model_name}')

        # Train KMeans model
        start_time = time.time()
        kmeans = KMeans(n_clusters=k, random_state=42, n_init="auto").fit(embedding_list)
        kmeans_time = time.time() - start_time
        print(f'Time cost of kmeans: {kmeans_time:.2f}s')

        # Calculate Calinski-Harabasz score
        start_time = time.time()
        ch = calinski_harabasz_score(embedding_list, kmeans.labels_)
        ch_time = time.time() - start_time
        print(f'Time cost of Calinski-Harabasz: {ch_time:.2f}s')

        # Calculate distances within clusters
        start_time = time.time()
        distances = []
        for i in range(k):
            code_cur_cluster = np.array(icd_code_list)[kmeans.labels_ == i]
            distance_temp = []

            # Skip empty clusters or clusters with only one element
            if len(code_cur_cluster) <= 1:
                distances.append(0)
                continue

            # Calculate pairwise distances within the cluster
            for code_1_idx in range(len(code_cur_cluster) - 1):
                for code_2_idx in range(code_1_idx + 1, len(code_cur_cluster)):
                    code1 = code_cur_cluster[code_1_idx]
                    code2 = code_cur_cluster[code_2_idx]

                    # Calculate the distance using our function
                    distance = calculate_icd_distance(code1, code2)
                    distance_temp.append(distance)

            # Calculate the average distance for this cluster
            distances.append(np.mean(distance_temp) if distance_temp else 0)

        distance_time = time.time() - start_time
        print(f'Time cost of calculating distance: {distance_time:.2f}s')

        # Save results
        save_path = os.path.join(output_dir, f'icd_result_clusters{k}.pkl')
        res = {
            'k': k,
            'kmeans_time': kmeans_time,
            'ch_time': ch_time,
            'distance_time': distance_time,
            'ch': ch,
            'distance': distances,
        }
        pd.to_pickle(res, save_path)
        print(f'Results saved to {save_path}')

def process_all_results():
    """Process results from all models and generate performance table"""
    print("Processing results from all models...")

    # Get list of all models
    model_names = BERT_MODELS + LLM_MODELS + EMBEDDING_MODELS

    # Get list of all k values processed
    ks = [int(k) for k in args.ks.split(',')]

    # Initialize performance table
    performance_table = {'model': [], 'k': [], 'ch': [], 'average_distance': []}

    # Process each model
    for model in model_names:
        model_dir = Path(f'logs/icd/{model}')
        if not model_dir.exists():
            print(f"Skipping model {model} - directory not found")
            continue

        for k in ks:
            result_path = os.path.join(model_dir, f'icd_result_clusters{k}.pkl')
            if not os.path.exists(result_path):
                print(f"Skipping k={k} for model {model} - result file not found")
                continue

            # Load and process result
            result = pd.read_pickle(result_path)
            performance_table['model'].append(model)
            performance_table['k'].append(k)
            performance_table['ch'].append(result['ch'])

            # Calculate average distance across all clusters
            valid_distances = [d for d in result['distance'] if d > 0]
            ave_dis = sum(valid_distances) / len(valid_distances) if valid_distances else 0
            performance_table['average_distance'].append(ave_dis)

    # Create DataFrame and save to CSV
    df = pd.DataFrame(performance_table)
    output_path = 'logs/icd/icd_performance.csv'
    df.to_csv(output_path, index=False)
    print(f"Performance table saved to {output_path}")

if __name__ == '__main__':
    # Parse k values
    k_values = [int(k) for k in args.ks.split(',')]

    if args.process_all:
        # First, run clustering for any models that haven't been processed
        for model in BERT_MODELS + LLM_MODELS + EMBEDDING_MODELS:
            run_clustering(model, k_values)

        # Then generate the performance table
        process_all_results()
    else:
        # Just process the specified model
        run_clustering(args.model, k_values)