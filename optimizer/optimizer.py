# python optimizer.py --all_candidates True --key_candidates 182,556,160 --optimization conceptNumber --dataset Youtube-8M_seg --repeat 10 --print True

import time
import pandas as pd
import os
import numpy as np
import random
import copy
import pickle
from mlxtend.preprocessing import TransactionEncoder
from torch.utils.data import DataLoader
from sklearn.metrics.pairwise import cosine_similarity
from external_info.conceptnet import hits_prob
import networkx as nx
from nltk.corpus import wordnet
import gensim
import argparse
from neural_network.neural_network import *
from neural_network.predict import predict_prob_nn
from external_info.conceptnet import *
from conditional_prob.train_prob import train_prob
from conditional_prob.test_prob import test_prob
from conditional_prob.trainval_prob import trainval_prob
from conditional_prob.train_NN_predict import predict_prob_train
from combine.kbc.kbc.predict import kbc_predict
from combine.combine_NN_predict_2 import predict_prob_combine_2
from combine.combine_NN_predict_3 import predict_prob_combine_3
from combine.conceptnet_NN_predict import predict_prob_combineconcept

def construct_dataset(train_dir, test_dir, test_true_dir, key_candidates):
    '''
    Construct the dataset in the form of [true_label] list. If the desired object exists in the video, true_label is 1. 
    '''
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

    # test_dataset = []
    # test_labels = []
    # for test_label in os.listdir(test_dir):
    #     test_labels.append(test_label)
    #     label_path = test_dir + test_label
    #     df = pd.read_csv(label_path)

    #     test_object = []
    #     for i in range(len(df["objects"])):
    #         if type(df["objects"][i]) != np.float64 and type(df["objects"][i]) != float:
    #             obj_res = df["objects"][i].split(',')
    #             for obj in obj_res:
    #                 if obj.split(':')[0] not in test_object and obj.split(':')[0] in te.columns_:
    #                     test_object.append(obj.split(':')[0])
    #     test_dataset.append(test_object)

    with open("../temp_result/finetune_videos.txt", 'rb') as f:
        finetune_videos = pickle.load(f)
    test_dataset = []
    test_labels = []
    for test_label in os.listdir(test_dir):
        if test_label not in finetune_videos:
            test_labels.append(test_label)
            label_path = test_dir + test_label
            df = pd.read_csv(label_path)

            test_object = []
            for i in range(len(df["objects"])):
                if type(df["objects"][i]) != np.float64 and type(df["objects"][i]) != float:
                    obj_res = df["objects"][i].split(',')
                    for obj in obj_res:
                        if obj.split(':')[0] not in test_object and obj.split(':')[0] in te.columns_:
                            test_object.append(obj.split(':')[0])
            test_dataset.append(test_object)

    test_true_dataset = []
    for test_true_label in test_labels:
        label_path = test_true_dir + test_true_label
        df = pd.read_csv(label_path)

        true_object = []
        for i in range(len(df["objects"])):
            if type(df["objects"][i]) != np.float64 and type(df["objects"][i]) != float:
                obj_res = df["objects"][i].split(',')
                for obj in obj_res:
                    if obj.split(':')[0] not in true_object and obj.split(':')[0] in te.columns_:
                        true_object.append(obj.split(':')[0])
        test_true_dataset.append(true_object)
    test_true_array = te.transform(test_true_dataset)

    if key_candidates == None:
        numberbatch_model = gensim.models.KeyedVectors.load_word2vec_format('/z/wenjiah/query_commonsense/data/ConceptNet/numberbatch-en-19.08.txt', binary=False)
        obj_freq = {}
        for true_object in test_true_dataset:
            for obj in true_object:
                if obj in obj_freq:
                    obj_freq[obj] += 1
                else:
                    obj_freq[obj] = 1

        key_candidates = []
        for i in range(len(te.columns_)):
            if te.columns_[i] in obj_freq and obj_freq[te.columns_[i]] >= 10 and te.columns_[i].replace(" ", "_") in numberbatch_model:
                key_candidates.append(i)

    dataset_list = []
    for i in range(len(key_candidates)):
        target_key = key_candidates[i]
        true_labels = []
        for j in range(len(test_true_array)):
            if test_true_array[j][target_key] == True:
                true_labels.append([1])
            else:
                true_labels.append([0])
        dataset_list.append(true_labels)
    
    return train_dataset, test_dataset, test_true_dataset, te, dataset_list, key_candidates, test_labels

def video_pred_prob(optimization, train_dir, val_dir, test_dir, test_dataset, test_true_dataset, te, dataset_list, key_candidates, dataset_name):
    '''
    Apply a classifier to predict the existence probability of the desired object in a video clip.
    Then each item is in the form of [true_label, pred_prob]
    '''
    if optimization == 'NN':
        for i in range(len(key_candidates)):
            target_key = key_candidates[i]
            test_dataloader = DataLoader(CustomDatasetTargetKey(test_dir, te, target_key), batch_size=64, shuffle=False)
            num_feature = len(te.columns_)
            num_class = len(te.columns_)
            dropout = 0.3
            model = DNN(num_feature, num_class, dropout)
            prediction_prob, _ = predict_prob_nn(test_dataloader=test_dataloader, model=model, model_dir="NN_model.pt")

            for j in range(len(dataset_list[i])):
                dataset_list[i][j].append(prediction_prob[j][target_key])

    elif optimization == 'conceptnet':
        for i in range(len(key_candidates)):
            target_key = key_candidates[i]
            target_obj = te.columns_[target_key]
            max_len = 2
            graph = nx.read_gpickle(target_obj.replace(" ", "_")+"_len"+str(max_len)+".gpickle")
            hubs, authorities = hits_prob(graph)

            for j in range(len(dataset_list[i])):
                test_object = test_dataset[j]
                video_prob = 1
                for obj in test_object:
                    if obj != te.columns_[target_key] and obj in list(graph.nodes):
                        video_prob *= 1-hubs[obj]
                video_prob = 1-video_prob
                dataset_list[i][j].append(video_prob)

    elif optimization == 'conceptNumber':
        numberbatch_model = gensim.models.KeyedVectors.load_word2vec_format('/z/wenjiah/query_commonsense/data/ConceptNet/numberbatch-en-19.08.txt', binary=False)

        start_time_1 = time.time()
        for i in range(len(key_candidates)):
            target_key = key_candidates[i]

            for j in range(len(dataset_list[i])):
                test_object = test_dataset[j]
                video_prob = 1
                for obj in test_object:
                    if obj != te.columns_[target_key] and obj.replace(" ", "_") in numberbatch_model:
                        video_prob *= 1-max(0, numberbatch_model.similarity(te.columns_[target_key].replace(" ", "_"), obj.replace(" ", "_")))
                video_prob = 1-video_prob
                dataset_list[i][j].append(video_prob)
        print("prob prediction time: ", time.time()-start_time_1)

    elif optimization == 'combine':
        # combine ConceptNet numberbatch and conditional prob in training data through a NN (standard)
        numberbatch_model = gensim.models.KeyedVectors.load_word2vec_format('/z/wenjiah/query_commonsense/data/ConceptNet/numberbatch-en-19.08.txt', binary=False)
        with open("../temp_result/train_cond_prob_dic_Youtube.txt", 'rb') as f:
        # with open("../temp_result/train_cond_prob_dic_10video_Youtube.txt", 'rb') as f:
        # with open("../temp_result/train_cond_prob_dic_50video_Youtube.txt", 'rb') as f:
        # with open("../temp_result/train_cond_prob_dic_100video_Youtube.txt", 'rb') as f:
        # with open("../temp_result/train_cond_prob_dic_20video_Youtube.txt", 'rb') as f:
            train_cond_prob_dic = pickle.load(f)

        start_time_1 = time.time()
        for i in range(len(key_candidates)):
            target_key = key_candidates[i]
            # prediction_prob = predict_prob_combine_2(train_dir=train_dir, test_dataset=test_dataset, te=te, target_key=target_key, numberbatch_model=numberbatch_model, model_dir="../temp_result/combine_model_2.pt", train_cond_prob_dic=train_cond_prob_dic)
            # prediction_prob = predict_prob_combine_2(train_dir=train_dir, test_dataset=test_dataset, te=te, target_key=target_key, numberbatch_model=numberbatch_model, model_dir="../temp_result/combine_model_2_10video.pt", train_cond_prob_dic=train_cond_prob_dic)
            # prediction_prob = predict_prob_combine_2(train_dir=train_dir, test_dataset=test_dataset, te=te, target_key=target_key, numberbatch_model=numberbatch_model, model_dir="../temp_result/combine_model_2_50video.pt", train_cond_prob_dic=train_cond_prob_dic)
            # prediction_prob = predict_prob_combine_2(train_dir=train_dir, test_dataset=test_dataset, te=te, target_key=target_key, numberbatch_model=numberbatch_model, model_dir="../temp_result/combine_model_2_100video.pt", train_cond_prob_dic=train_cond_prob_dic)
            # prediction_prob = predict_prob_combine_2(train_dir=train_dir, test_dataset=test_dataset, te=te, target_key=target_key, numberbatch_model=numberbatch_model, model_dir="../temp_result/combine_model_2_20video.pt", train_cond_prob_dic=train_cond_prob_dic)
            # prediction_prob = predict_prob_combine_2(train_dir=train_dir, test_dataset=test_dataset, te=te, target_key=target_key, numberbatch_model=numberbatch_model, model_dir="../temp_result/combine_model_2_50video_20val.pt", train_cond_prob_dic=train_cond_prob_dic)
            # prediction_prob = predict_prob_combine_3(train_dir=train_dir, test_dataset=test_dataset, te=te, target_key=target_key, numberbatch_model=numberbatch_model, model_dir="../temp_result/combine_model_3_100video_20val.pt", train_cond_prob_dic=train_cond_prob_dic)
            prediction_prob = predict_prob_combine_3(train_dir=train_dir, test_dataset=test_dataset, te=te, target_key=target_key, numberbatch_model=numberbatch_model, model_dir="../temp_result/combine_model_3_finetune_key89.pt", train_cond_prob_dic=train_cond_prob_dic)
            for j in range(len(dataset_list[i])):
                dataset_list[i][j].append(prediction_prob[j])
        print("prob prediction time: ", time.time()-start_time_1)

    elif optimization == 'combineConcept':
        # combine; only ConceptNet numberbatch in the input 
        numberbatch_model = gensim.models.KeyedVectors.load_word2vec_format('/z/wenjiah/query_commonsense/data/ConceptNet/numberbatch-en-19.08.txt', binary=False)

        start_time_1 = time.time()
        for i in range(len(key_candidates)):
            target_key = key_candidates[i]
            # prediction_prob = predict_prob_combineconcept(test_dataset=test_dataset, te=te, target_key=target_key, numberbatch_model=numberbatch_model, model_dir="../temp_result/combineconcept_model.pt")
            # prediction_prob = predict_prob_combineconcept(test_dataset=test_dataset, te=te, target_key=target_key, numberbatch_model=numberbatch_model, model_dir="../temp_result/combineconcept_model_100video_20val.pt")
            prediction_prob = predict_prob_combineconcept(test_dataset=test_dataset, te=te, target_key=target_key, numberbatch_model=numberbatch_model, model_dir="../temp_result/combineconcept_model_finetune_key89.pt")
            for j in range(len(dataset_list[i])):
                dataset_list[i][j].append(prediction_prob[j])
        print("prob prediction time: ", time.time()-start_time_1)

    elif optimization == 'trainNN':
        with open("../temp_result/train_cond_prob_dic_Youtube.txt", 'rb') as f:
        # with open("../temp_result/train_cond_prob_dic_10video_Youtube.txt", 'rb') as f:
        # with open("../temp_result/train_cond_prob_dic_50video_Youtube.txt", 'rb') as f:
        # with open("../temp_result/train_cond_prob_dic_100video_Youtube.txt", 'rb') as f:
        # with open("../temp_result/train_cond_prob_dic_20video_Youtube.txt", 'rb') as f:
            train_cond_prob_dic = pickle.load(f)

        start_time_1 = time.time()
        for i in range(len(key_candidates)):
            target_key = key_candidates[i]
            # prediction_prob = predict_prob_train(train_dir=train_dir, test_dataset=test_dataset, te=te, target_key=target_key, model_dir="../temp_result/training_model_1.pt", train_cond_prob_dic=train_cond_prob_dic)
            # prediction_prob = predict_prob_train(train_dir=train_dir, test_dataset=test_dataset, te=te, target_key=target_key, model_dir="../temp_result/training_model_10video.pt", train_cond_prob_dic=train_cond_prob_dic)
            # prediction_prob = predict_prob_train(train_dir=train_dir, test_dataset=test_dataset, te=te, target_key=target_key, model_dir="../temp_result/training_model_50video.pt", train_cond_prob_dic=train_cond_prob_dic)
            # prediction_prob = predict_prob_train(train_dir=train_dir, test_dataset=test_dataset, te=te, target_key=target_key, model_dir="../temp_result/training_model_100video.pt", train_cond_prob_dic=train_cond_prob_dic)
            # prediction_prob = predict_prob_train(train_dir=train_dir, test_dataset=test_dataset, te=te, target_key=target_key, model_dir="../temp_result/training_model_20video.pt", train_cond_prob_dic=train_cond_prob_dic)
            # prediction_prob = predict_prob_train(train_dir=train_dir, test_dataset=test_dataset, te=te, target_key=target_key, model_dir="../temp_result/training_model_100video_20val.pt", train_cond_prob_dic=train_cond_prob_dic)
            prediction_prob = predict_prob_train(train_dir=train_dir, test_dataset=test_dataset, te=te, target_key=target_key, model_dir="../temp_result/training_model_1_finetune_key89.pt", train_cond_prob_dic=train_cond_prob_dic)
            for j in range(len(dataset_list[i])):
                dataset_list[i][j].append(prediction_prob[j])
        print("prob prediction time: ", time.time()-start_time_1)

    elif optimization == 'trainval':
        with open("../temp_result/trainval_cond_prob_dic_Youtube.txt", 'rb') as f:
        # with open("../temp_result/trainval_cond_prob_dic_10video_Youtube.txt", 'rb') as f:
        # with open("../temp_result/trainval_cond_prob_dic_20video_Youtube.txt", 'rb') as f:
        # with open("../temp_result/trainval_cond_prob_dic_50video_Youtube.txt", 'rb') as f:
        # with open("../temp_result/trainval_cond_prob_dic_100video_Youtube.txt", 'rb') as f:
        # with open("../temp_result/trainval_cond_prob_dic_100video_20val_Youtube.txt", 'rb') as f:
            trainval_cond_prob_dic = pickle.load(f)

        for i in range(len(key_candidates)):
            target_key = key_candidates[i]
            
            for j in range(len(dataset_list[i])):
                test_object = test_dataset[j]
                video_prob = 1
                for obj in test_object:
                    if obj != te.columns_[target_key]:
                        video_prob *= 1-trainval_cond_prob_dic[te.columns_[target_key]][obj]
                video_prob = 1-video_prob
                dataset_list[i][j].append(video_prob)

    elif optimization == 'test':
        for i in range(len(key_candidates)):
            target_key = key_candidates[i]
            test_cond_prob = test_prob(test_dataset=test_dataset, test_true_dataset=test_true_dataset, te=te, target_key=target_key)
            
            for j in range(len(dataset_list[i])):
                test_object = test_dataset[j]
                video_prob = 1
                for obj in test_object:
                    if obj != te.columns_[target_key]:
                        video_prob *= 1-test_cond_prob[obj]
                video_prob = 1-video_prob
                dataset_list[i][j].append(video_prob)

    elif optimization == 'perfect':
        start_time_1 = time.time()
        for i in range(len(key_candidates)):
            for j in range(len(dataset_list[i])):
                dataset_list[i][j].append(dataset_list[i][j][0])
        print("prob prediction time: ", time.time()-start_time_1)

    elif optimization == 'random':
        start_time_1 = time.time()
        for i in range(len(key_candidates)):
            for j in range(len(dataset_list[i])):
                dataset_list[i][j].append(random.random())
        print("prob prediction time: ", time.time()-start_time_1)

    else:
        raise NameError('Wrong optimization name')

    return dataset_list

def process(dataset_list):
    '''
    Process the videos according to the order of pred_prob.
    TODO: Handle items with high or low pred_prob
    '''
    start_time_2 = time.time()

    process_num_list = []
    for i in range(len(dataset_list)):
        dataset_labels = dataset_list[i]
        cur_labels = copy.deepcopy(dataset_labels)
        num_true = sum([data[0] for data in cur_labels])
        cur_labels.sort(reverse = True, key = lambda data:data[1])
        limits = np.arange(0.1, 1.01, 0.1) * num_true

        process_num = []
        for limit in limits:
            res_size = 0
            process_idx = 0
            while res_size < limit:
                next_item = cur_labels[process_idx]
                if next_item[0] == 1:
                    res_size += 1
                process_idx += 1
            process_num.append(process_idx)
        process_num_list.append(process_num)

    print("time of selecting video candidates for all the LIMIT settings: ", time.time()-start_time_2)
    
    return process_num_list

def processed_labels(dataset_list, test_dataset, test_true_dataset, te, key_candidates, test_labels):
    '''
    Print out video clips in the processing order
    '''
    processed_videos_list = []
    for i in range(len(dataset_list)):
        dataset_labels = dataset_list[i]
        cur_labels = copy.deepcopy(dataset_labels)
        for j in range(len(cur_labels)):
            cur_labels[j].append(j)
        cur_labels.sort(reverse = True, key = lambda data:data[1])

        processed_videos = []
        for j in range(len(cur_labels)):
            processed_video = test_dataset[cur_labels[j][2]]
            processed_video_true = test_true_dataset[cur_labels[j][2]]
            video_id = test_labels[cur_labels[j][2]].split('.')[0]
            processed_str = ""
            for k in range(len(processed_video)):
                if processed_video[k] != te.columns_[key_candidates[i]]:
                    processed_str += processed_video[k] + ", "
            processed_true_str = ""
            for k in range(len(processed_video_true)):
                processed_true_str += processed_video_true[k] + ", "
            processed_videos.append([cur_labels[j][1], processed_str[:-2], processed_true_str[:-2], video_id])
        processed_videos_list.append(processed_videos)

    return processed_videos_list

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--all_candidates', required=True)
    parser.add_argument('--key_candidates')
    parser.add_argument('--optimization', required=True)
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--repeat', type=int, required=True)
    parser.add_argument('--print', required=True)
    args = parser.parse_args()

    all_candidates = args.all_candidates
    if all_candidates == 'True':
        key_candidates = None
    else:
        key_candidates = args.key_candidates
        key_candidates = [int(item) for item in key_candidates.split(',')]
    optimization = args.optimization
    dataset_name = args.dataset
    repeat_times = args.repeat
    print_videos = args.print

    train_dir = "/z/wenjiah/query_commonsense/data/"+dataset_name+"/video_label/yolo-9000-clip-select-split/train/"
    val_dir = "/z/wenjiah/query_commonsense/data/"+dataset_name+"/video_label/yolo-9000-clip-select-split/val/"
    test_dir = "/z/wenjiah/query_commonsense/data/"+dataset_name+"/video_label/yolo-9000-clip-select-split/test/"
    test_true_dir = "/z/wenjiah/query_commonsense/data/"+dataset_name+"/video_label/yolo-9000-clip-select-split/test_true/"

    train_dataset, test_dataset, test_true_dataset, te, dataset_list, key_candidates, test_labels = construct_dataset(train_dir, test_dir, test_true_dir, key_candidates)
    process_num_avg_list = np.zeros([len(key_candidates), 10])
    for trial in range(repeat_times):
        predicted_dataset_list = video_pred_prob(optimization, train_dir, val_dir, test_dir, test_dataset, test_true_dataset, te, copy.deepcopy(dataset_list), key_candidates, dataset_name)
        process_num_list = process(predicted_dataset_list)
        process_num_avg_list += np.array(process_num_list)

        if trial == 0 and print_videos == "True":
            processed_videos_list = processed_labels(predicted_dataset_list, test_dataset, test_true_dataset, te, key_candidates, test_labels)
            for i in range(len(key_candidates)):
                processed_videos = processed_videos_list[i]
                df = pd.DataFrame(processed_videos, columns=["Prediction score", "Approximate labels", "Ground truth", "Video ID"])
                df.to_csv("../temp_result/"+"processed_videos_"+str(key_candidates[i])+".csv", index=None)
    process_num_avg_list /= repeat_times        

    store_dict = {}
    store_dict["process num"] = process_num_avg_list
    store_dict["keys"] = key_candidates
    key_names = []
    for i in range(len(key_candidates)):
        key_names.append(te.columns_[key_candidates[i]])
    store_dict["key names"] = key_names
    with open("../../result/raw_result/opt_Youtube_"+optimization+"_finetune_key89_all.txt", 'wb') as f:
    # with open("../../result/raw_result/opt_Youtube_"+optimization+"_100video_20val_all.txt", 'wb') as f:
    # with open("../temp_result/opt_"+optimization+".txt", 'wb') as f:
    # with open("../temp_result/opt_combine_cond_NN_2.txt", 'wb') as f:
    # with open("opt_4%train_all.txt", 'wb') as f:
    # with open("opt_train_frame%2_all.txt", 'wb') as f:
        pickle.dump(store_dict, f)