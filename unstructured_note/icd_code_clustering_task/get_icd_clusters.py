import os
import jsonlines
import numpy as np
import pandas as pd
import time
import argparse

from sklearn.cluster import KMeans
from sklearn.metrics import calinski_harabasz_score
from scipy.spatial.distance import cdist
from scipy.spatial.distance import pdist
from multiprocessing.pool import ThreadPool as Pool

from utils.build_tree import find_distance, build_tree_fun

parser = argparse.ArgumentParser()
parser.description = 'please enter parameter model name'
parser.add_argument('--model', type=str, default='OpenBioLLM')
args = parser.parse_args()

model_name = args.model


def cal_dist(code1, code2):
    dis = find_distance(tree, code1, code2)
    return dis


def worker(k):
    print('-' * 10, 'k = {}'.format(k), '-' * 10)
    # 构建并训练模型
    start_time = time.time()
    kmeans = KMeans(n_clusters=k, random_state=42, n_init="auto").fit(DATA)
    kmeans_time = time.time() - start_time
    print('time cost of kmeans: {}s'.format(kmeans_time))

    start_time = time.time()
    ch = calinski_harabasz_score(DATA, kmeans.labels_)
    ch_time = time.time() - start_time
    print('time cost of calinski-harabasz: {}s'.format(ch_time))

    start_time = time.time()
    # using double for
    distances = []
    for i in range(k):
        code_cur_cluster = np.array(CODE)[kmeans.labels_ == i]
        distance_temp = []
        for code_1 in range(len(code_cur_cluster) - 1):
            for code_2 in range(code_1 + 1, len(code_cur_cluster)):
                # using a simple method that result is 2 or 4
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
                # using a complex method that find distance from tree
                # distance_temp.append(find_distance(tree, code_cur_cluster[code_1], code_cur_cluster[code_2]))
                # print("Current: distance_temp",k, distance_temp)
        distances.append(np.mean(distance_temp))
    # using pdist
    # distances = []
    # for i in range(k):
    #     code_cur_cluster = np.array(CODE)[kmeans.labels_ == i][:, np.newaxis]
    #     dists = pdist(code_cur_cluster, metric=cal_dist)
    #     distances.append(np.mean(dists))
    distance_time = time.time() - start_time
    print('time cost of calculating distance: {}s'.format(distance_time))
    save_path = f'./logs/icd/{model_name}/icd_result_clusters{k}.pkl'
    res = {
            'k': k,
            'kmeans_time': kmeans_time,
            'ch_time': ch_time,
            'distance_time': distance_time,
            'ch': ch,
            'distance': distances,
        }
    print(res)
    pd.to_pickle(res, save_path)


pool_size = 10  # your "parallelness"

tree_path = r'datasets/icd10cm_order_2023.txt'
tree = build_tree_fun(tree_path)

embedding_path = f'./logs/icd/{model_name}/icd_embeddings.pkl'

DATA = []
CODE = []
# with open(embedding_path) as file:
#     for disease_info in jsonlines.Reader(file):
#         code = disease_info["code"]
#         embedding = disease_info["embedding"]
#         if len(code) <= 4:
#             CODE.append(code)
#             DATA.append(embedding)
embeddings = pd.read_pickle(embedding_path)
for disease_info in embeddings:
    code = disease_info["code"][0]
    embedding = disease_info["embedding"]
    if len(code) <= 4:
        CODE.append(code)
        DATA.append(embedding)

# pool = Pool(pool_size)
# # for k in [5, 10, 15, 20, 25, 30, 35, 40, 45, 50]:
# for k in [5,10]:
#     pool.apply_async(worker, (k,))
# pool.close()
# pool.join()


for k in [2, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50]:
    worker(k)
