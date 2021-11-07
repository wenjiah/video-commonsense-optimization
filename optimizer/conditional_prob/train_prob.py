import pandas as pd
import os
import numpy as np
import pickle
from mlxtend.preprocessing import TransactionEncoder

def train_prob(train_dir, frame_interval, video_interval, te, target_key):
    part_train_dataset = [] 
    for train_label in os.listdir(train_dir):
        label_path = train_dir + train_label
        df = pd.read_csv(label_path)

        detected_object = []
        for i in range(len(df["objects"])):
            if i%frame_interval == 0 and type(df["objects"][i]) != np.float64 and type(df["objects"][i]) != float:
                obj_res = df["objects"][i].split(',')
                for obj in obj_res:
                    if obj.split(':')[0] not in detected_object:
                        detected_object.append(obj.split(':')[0])
        part_train_dataset.append(detected_object)

    train_cond_prob = {}
    for obj in te.columns_:
        count_obj = 0
        count_tar = 0
        for k in range(len(part_train_dataset)):
            if k%video_interval == 0:
                train_object = part_train_dataset[k]
                if obj in train_object:
                    count_obj += 1
                    if te.columns_[target_key] in train_object:
                        count_tar += 1
        if count_obj > 0:
            train_cond_prob[obj] = count_tar/count_obj
        else:
            train_cond_prob[obj] = 0
    
    return train_cond_prob

if __name__ == "__main__":
    dataset_name = "Youtube-8M_seg"
    train_dir = "/z/wenjiah/query_commonsense/data/"+dataset_name+"/video_label/yolo-9000-clip-select-split/train/"

    train_dataset = []   # video-level objects
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

    train_cond_prob_dic = {}
    for key in range(len(te.columns_)):
        # train_cond_prob_dic[te.columns_[key]] = train_prob(train_dir=train_dir, frame_interval=1, video_interval=1, te=te, target_key=key)
        # train_cond_prob_dic[te.columns_[key]] = train_prob(train_dir=train_dir, frame_interval=1, video_interval=10, te=te, target_key=key)
        # train_cond_prob_dic[te.columns_[key]] = train_prob(train_dir=train_dir, frame_interval=1, video_interval=50, te=te, target_key=key)
        # train_cond_prob_dic[te.columns_[key]] = train_prob(train_dir=train_dir, frame_interval=1, video_interval=100, te=te, target_key=key)
        train_cond_prob_dic[te.columns_[key]] = train_prob(train_dir=train_dir, frame_interval=1, video_interval=20, te=te, target_key=key)

    # with open("../../temp_result/train_cond_prob_dic_Youtube.txt", 'wb') as f:
    # with open("../../temp_result/train_cond_prob_dic_10video_Youtube.txt", 'wb') as f:
    # with open("../../temp_result/train_cond_prob_dic_50video_Youtube.txt", 'wb') as f:
    # with open("../../temp_result/train_cond_prob_dic_100video_Youtube.txt", 'wb') as f:
    with open("../../temp_result/train_cond_prob_dic_20video_Youtube.txt", 'wb') as f:
        pickle.dump(train_cond_prob_dic, f)