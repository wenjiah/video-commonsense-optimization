import pandas as pd
import os
import numpy as np
import pickle
from mlxtend.preprocessing import TransactionEncoder

def trainval_prob(obj_freq, obj_pair_freq, te, target_key):
    trainval_cond_prob = {}
    for obj_key in range(len(te.columns_)):
        if obj_freq[obj_key] > 0:
            trainval_cond_prob[te.columns_[obj_key]] = obj_pair_freq[obj_key][target_key]/obj_freq[obj_key]
        else:
            trainval_cond_prob[te.columns_[obj_key]] = 0
    
    return trainval_cond_prob

if __name__ == "__main__":
    dataset_name = "Youtube-8M_seg"
    train_dir = "/z/wenjiah/query_commonsense/data/"+dataset_name+"/video_label/yolo-9000-clip-select-split/train/"
    val_dir = "/z/wenjiah/query_commonsense/data/"+dataset_name+"/video_label/yolo-9000-clip-select-split/val/"
    test_true_dir = "/z/wenjiah/query_commonsense/data/"+dataset_name+"/video_label/yolo-9000-clip-select-split/test_true/"

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
    train_video_interval = 1
    val_frame_interval = 1
    # val_video_interval = 1
    val_video_interval = 1

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
                    if obj.split(':')[0] not in detected_object and obj.split(':')[0] in te.columns_:
                        detected_object.append(obj.split(':')[0])
        part_val_dataset.append(detected_object)

    with open("../../temp_result/finetune_videos.txt", 'rb') as f:
        finetune_videos = pickle.load(f)
    test_true_dataset = []
    for test_true_label in finetune_videos:
        label_path = test_true_dir + test_true_label
        df = pd.read_csv(label_path)

        test_true_object = []
        for i in range(len(df["objects"])):
            if type(df["objects"][i]) != np.float64 and type(df["objects"][i]) != float:
                obj_res = df["objects"][i].split(',')
                for obj in obj_res:
                    if obj.split(':')[0] not in test_true_object and obj.split(':')[0] in te.columns_:
                        test_true_object.append(obj.split(':')[0])
        test_true_dataset.append(test_true_object)
    
    part_dataset = part_train_dataset[::train_video_interval] + part_val_dataset[::val_video_interval] + test_true_dataset
    part_array = te.transform(part_dataset)

    obj_freq = np.zeros(len(te.columns_))
    obj_pair_freq = np.zeros([len(te.columns_),len(te.columns_)])

    for obj_array in part_array:
        for i in range(len(te.columns_)):
            if obj_array[i] == True:
                obj_freq[i] += 1
        for i in range(len(te.columns_)):
            for j in range(len(te.columns_)):
                if obj_array[i] == True and obj_array[j] == True:
                    obj_pair_freq[i][j] += 1

    trainval_cond_prob_dic = {}
    for key in range(len(te.columns_)):
        trainval_cond_prob_dic[te.columns_[key]] = trainval_prob(obj_freq=obj_freq, obj_pair_freq=obj_pair_freq, te=te, target_key=key)

    with open("../../temp_result/trainval_cond_prob_dic_finetune_Youtube.txt", 'wb') as f:
        pickle.dump(trainval_cond_prob_dic, f)