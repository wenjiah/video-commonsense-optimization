import time
import pandas as pd
import os
import numpy as np
import random
import copy
import pickle
from mlxtend.preprocessing import TransactionEncoder
import argparse
import cv2
from scipy.spatial.distance import hamming
import json

train_dir = "video_label/yolo-9000-clip-select-split/train/"
test_dir = "video_label/yolo-9000-clip-select-split/test/"

train_dataset = []
for train_label in os.listdir(train_dir):
    label_path = train_dir + train_label
    df = pd.read_csv(label_path)

    detected_object = []
    for i in range(len(df["objects"])):
        if type(df["objects"][i]) != np.float64 and type(df["objects"][i]) != float:
            obj_res = df["objects"][i].split(',')
            for obj in obj_res:
                if obj.split(':')[0] not in detected_object:
                    detected_object.append(obj.split(':')[0])
    train_dataset.append(detected_object)

te = TransactionEncoder()
te = te.fit(train_dataset)

random.seed(0)
with open("../../model/finetune_videos_youtube.txt", 'rb') as f:
    finetune_videos = pickle.load(f)
test_dataset = []
test_labels = []
for test_label in os.listdir(test_dir):
    if test_label not in finetune_videos:
        test_labels.append(test_label)
        label_path = test_dir + test_label
        df = pd.read_csv(label_path)

        test_object = []
        for i in sorted(random.sample(range(len(df["objects"])), int(len(df["objects"])))): 
            if type(df["objects"][i]) != np.float64 and type(df["objects"][i]) != float:
                obj_res = df["objects"][i].split(',')
                for obj in obj_res:
                    if obj.split(':')[0] not in test_object and obj.split(':')[0] in te.columns_:
                        test_object.append(obj.split(':')[0])
        test_dataset.append(test_object)
test_array = te.transform(test_dataset)

start_time = time.time()

distance = 30
cluster_id = 0
clusters = {} # key: cluster ID; pair: [video index, centroid]

for i in range(len(test_array)):
    if cluster_id == 0:
        clusters[cluster_id] = [[i], test_array[i]]
        cluster_id += 1
        continue

    min_distance = float('inf')
    min_cluster_id = None
    for key in clusters.keys():
        centroid = clusters[key][1]
        cur_distance = hamming(test_array[i], centroid) * len(centroid)
        if cur_distance < min_distance:
            min_distance = cur_distance
            min_cluster_id = key

    if min_distance <= distance:
        clusters[min_cluster_id][0].append(i)
        clusters[min_cluster_id][1] = [clusters[min_cluster_id][1][j] or test_array[i][j] for j in range(len(test_array[i]))]
    else:
        clusters[cluster_id] = [[i], test_array[i]]
        cluster_id += 1

print("number of clusters:", cluster_id-1)
print("cluster time:", time.time()-start_time)

for key in clusters.keys():
    clusters[key][1] = [int(centroid_val) for centroid_val in clusters[key][1]]

with open("index_cluster_youtube.json", "w") as f:
    json.dump(clusters, f)