import pandas as pd
import os
import numpy as np
import pickle
from mlxtend.preprocessing import TransactionEncoder

def trainval_prob(part_dataset, te, target_key):
    trainval_cond_prob = {}
    for obj in te.columns_:
        count_obj = 0
        count_tar = 0
        for k in range(len(part_dataset)):
            part_object = part_dataset[k]
            if obj in part_object:
                count_obj += 1
                if te.columns_[target_key] in part_object:
                    count_tar += 1
        if count_obj > 0:
            trainval_cond_prob[obj] = count_tar/count_obj
        else:
            trainval_cond_prob[obj] = 0
    
    return trainval_cond_prob

if __name__ == "__main__":
    dataset_name = "Youtube-8M_seg"
    train_dir = "/z/wenjiah/query_commonsense/data/"+dataset_name+"/video_label/yolo-9000-clip-select-split/train/"
    val_dir = "/z/wenjiah/query_commonsense/data/"+dataset_name+"/video_label/yolo-9000-clip-select-split/val/"

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

    train_frame_interval = 1
    # train_video_interval = 1
    train_video_interval = 100
    val_frame_interval = 1
    # val_video_interval = 1
    val_video_interval = 20

    part_train_dataset = [] 
    for train_label in os.listdir(train_dir):
        label_path = train_dir + train_label
        df = pd.read_csv(label_path)

        detected_object = []
        for i in range(len(df["objects"])):
            if i%train_frame_interval == 0 and type(df["objects"][i]) != np.float64 and type(df["objects"][i]) != float:
                obj_res = df["objects"][i].split(',')
                for obj in obj_res:
                    if obj.split(':')[0] not in detected_object:
                        detected_object.append(obj.split(':')[0])
        part_train_dataset.append(detected_object)

    part_val_dataset = []
    for val_label in os.listdir(val_dir):
        label_path = val_dir + val_label
        df = pd.read_csv(label_path)

        detected_object = []
        for i in range(len(df["objects"])):
            if i%val_frame_interval == 0 and type(df["objects"][i]) != np.float64 and type(df["objects"][i]) != float:
                obj_res = df["objects"][i].split(',')
                for obj in obj_res:
                    if obj.split(':')[0] not in detected_object:
                        detected_object.append(obj.split(':')[0])
        part_val_dataset.append(detected_object)
    
    part_dataset = part_train_dataset[::train_video_interval] + part_val_dataset[::val_video_interval]

    trainval_cond_prob_dic = {}
    for key in range(len(te.columns_)):
        trainval_cond_prob_dic[te.columns_[key]] = trainval_prob(part_dataset=part_dataset, te=te, target_key=key)

    # with open("../../temp_result/trainval_cond_prob_dic_Youtube.txt", 'wb') as f:
    # with open("../../temp_result/trainval_cond_prob_dic_10video_Youtube.txt", 'wb') as f:
    # with open("../../temp_result/trainval_cond_prob_dic_20video_Youtube.txt", 'wb') as f:
    # with open("../../temp_result/trainval_cond_prob_dic_50video_Youtube.txt", 'wb') as f:
    # with open("../../temp_result/trainval_cond_prob_dic_100video_Youtube.txt", 'wb') as f:
    with open("../../temp_result/trainval_cond_prob_dic_100video_20val_Youtube.txt", 'wb') as f:
        pickle.dump(trainval_cond_prob_dic, f)