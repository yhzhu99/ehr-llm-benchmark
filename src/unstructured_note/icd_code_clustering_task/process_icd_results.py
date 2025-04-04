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

from unstructured_note.utils.build_tree import find_distance, build_tree_fun
from unstructured_note.utils.config import MODELS_CONFIG

# Create model lists
BERTBasedModels = [model["model_name"] for model in MODELS_CONFIG if model["model_type"] == "BERT"]
LLM = [model["model_name"] for model in MODELS_CONFIG if model["model_type"] == "GPT"]

parser = argparse.ArgumentParser(description='Process ICD embeddings with clustering')
parser.add_argument('--model', type=str, default='BERT', choices=BERTBasedModels + LLM)
parser.add_argument('--process_all', action='store_true', help='Process all models and generate performance table')
parser.add_argument('--ks', type=str, default='2,5,10,15,20,25,30,35,40,45,50',
                    help='Comma-separated list of cluster numbers')
args = parser.parse_args()

def cal_dist(code1, code2):
    """Calculate distance between two ICD codes using the tree"""
    dis = find_distance(tree, code1, code2)
    return dis

def run_clustering(model_name, k_values):
    """Run clustering with specified k values for a given model"""
    print(f"Running clustering for model: {model_name}")

    # Build ICD tree
    tree_path = 'my_datasets/icd10/icd10cm_order_2023.txt'
    global tree
    tree = build_tree_fun(tree_path)

    # Load embeddings
    embedding_path = f'logs/icd/{model_name}/icd_embeddings.pkl'
    if not os.path.exists(embedding_path):
        print(f"Embedding file not found: {embedding_path}")
        return

    embeddings = pd.read_pickle(embedding_path)

    # Extract codes and embeddings
    DATA = []
    CODE = []
    for disease_info in embeddings:
        code = disease_info["code"][0] if isinstance(disease_info["code"], list) else disease_info["code"]
        embedding = disease_info["embedding"]
        if len(code) <= 4:
            CODE.append(code)
            DATA.append(embedding)

    # Ensure output directory exists
    output_dir = Path(f'logs/icd/{model_name}')
    output_dir.mkdir(parents=True, exist_ok=True)

    # Process each k value
    for k in k_values:
        print(f'Processing k = {k} for model {model_name}')

        # Train KMeans model
        start_time = time.time()
        kmeans = KMeans(n_clusters=k, random_state=42, n_init="auto").fit(DATA)
        kmeans_time = time.time() - start_time
        print(f'Time cost of kmeans: {kmeans_time:.2f}s')

        # Calculate Calinski-Harabasz score
        start_time = time.time()
        ch = calinski_harabasz_score(DATA, kmeans.labels_)
        ch_time = time.time() - start_time
        print(f'Time cost of Calinski-Harabasz: {ch_time:.2f}s')

        # Calculate distances within clusters
        start_time = time.time()
        distances = []
        for i in range(k):
            code_cur_cluster = np.array(CODE)[kmeans.labels_ == i]
            distance_temp = []

            # Skip empty clusters
            if len(code_cur_cluster) <= 1:
                distances.append(0)
                continue

            for code_1 in range(len(code_cur_cluster) - 1):
                for code_2 in range(code_1 + 1, len(code_cur_cluster)):
                    # Simple distance calculation based on code structure
                    if code_cur_cluster[code_1][0] == code_cur_cluster[code_2][0]:
                        if len(code_cur_cluster[code_1]) == len(code_cur_cluster[code_2]) == 3:
                            distance_temp.append(2)
                        elif len(code_cur_cluster[code_1]) == len(code_cur_cluster[code_2]) == 4:
                            if code_cur_cluster[code_1][:3] == code_cur_cluster[code_2][:3]:
                                distance_temp.append(2)
                            else:
                                distance_temp.append(4)
                        else:
                            if code_cur_cluster[code_1][:3] == code_cur_cluster[code_2][:3]:
                                distance_temp.append(1)
                            else:
                                distance_temp.append(3)
                    else:
                        if len(code_cur_cluster[code_1]) == len(code_cur_cluster[code_2]) == 3:
                            distance_temp.append(4)
                        elif len(code_cur_cluster[code_1]) == len(code_cur_cluster[code_2]) == 4:
                            distance_temp.append(6)
                        else:
                            distance_temp.append(5)

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
    model_names = BERTBasedModels + LLM

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
        for model in BERTBasedModels + LLM:
            run_clustering(model, k_values)

        # Then generate the performance table
        process_all_results()
    else:
        # Just process the specified model
        run_clustering(args.model, k_values)