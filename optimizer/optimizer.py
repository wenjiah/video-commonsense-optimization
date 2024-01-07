# python optimizer.py --all_candidates True (--key_candidates 182,556,160) --optimization conceptNumber --dataset Youtube-8M_seg --repeat 1 --checkpoint finetune_youtube/checkpoint-1500 --index_percent 100 --result_name _finetune_all
# python optimizer.py --all_candidates True (--key_candidates 182,556,160) --optimization conceptNumber --dataset HowTo100M --repeat 1  --checkpoint finetune_How/checkpoint-1000 --index_percent 100 --result_name _finetune_all

import time
import pandas as pd
import os
import numpy as np
import random
import copy
import pickle
from mlxtend.preprocessing import TransactionEncoder
import gensim
import argparse
import cv2
import json
from BERT_YouTube.predict_youtube import predict_prob_bert
from prob_from_kg import cond_prob
from LLM_sort import *

def construct_dataset(train_dir, test_dir, test_true_dir, key_candidates, finetune_filename, index_percent):
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

    random.seed(0)
    with open("../model_prob_info/"+finetune_filename+".txt", 'rb') as f:
        finetune_videos = pickle.load(f)
    test_dataset = []
    test_labels = []
    for test_label in os.listdir(test_dir):
        if test_label not in finetune_videos:
            test_labels.append(test_label)
            label_path = test_dir + test_label
            df = pd.read_csv(label_path)

            test_object = []
            for i in sorted(random.sample(range(len(df["objects"])), int(len(df["objects"])*index_percent/100))): 
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
    
    return train_dataset, test_dataset, test_true_dataset, te, dataset_list, test_labels

def video_pred_prob(optimization, test_dataset, te, dataset_list, key_candidates, dataset_name, test_labels, test_dir, test_true_dir, test_true_frames_dir, checkpoint, prompt_style):
    '''
    Apply a classifier to predict the existence probability of the target object in a video clip.
    Then each item is in the form of [true_label, pred_prob]
    '''

    if optimization == 'conceptNumber':
        # old estimation with the independence assumption
        numberbatch_model = gensim.models.KeyedVectors.load_word2vec_format('../../data/ConceptNet/numberbatch-en-19.08.txt', binary=False)

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

    elif optimization == 'kg':
        # new estimation without independence assumption
        numberbatch_model = gensim.models.KeyedVectors.load_word2vec_format('../../data/ConceptNet/numberbatch-en-19.08.txt', binary=False)
        page_views_avg = pd.read_pickle(dataset_name+"_pageview.pkl")
        # page_views_prob = page_views_avg/max(page_views_avg)
        page_views_prob = np.sqrt(page_views_avg/max(page_views_avg))
        # page_views_prob = pow(page_views_avg/max(page_views_avg), 1/6)

        start_time_1 = time.time()
        for i in range(len(key_candidates)):
            target_key = key_candidates[i]

            for j in range(len(dataset_list[i])):
                test_object = test_dataset[j]
                cond_object = []
                for obj in test_object:
                    if obj != te.columns_[target_key] and obj.replace(" ", "_") in numberbatch_model:
                        cond_object.append(obj)
                video_prob = cond_prob(cond_object=cond_object, target_obj=[te.columns_[target_key]], embeddings=numberbatch_model, single_prob_df=page_views_prob)
                dataset_list[i][j].append(min(video_prob, 1))
        print("prob prediction time: ", time.time()-start_time_1)


    elif optimization == 'visual': 
        start_time_1 = time.time()
        for i in range(len(key_candidates)):
            target_key = key_candidates[i]
            prediction_prob = predict_prob_bert(test_dataset=test_dataset, te=te, target_key=target_key, model_dir="../model_prob_info/"+checkpoint)
            for j in range(len(dataset_list[i])):
                dataset_list[i][j].append(prediction_prob[j])
        print("prob prediction time: ", time.time()-start_time_1)

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
    
    elif optimization == "llmEZ":
        start_time_1 = time.time()
        for i in range(len(key_candidates)):
            target_key = key_candidates[i]
            target_obj = te.columns_[target_key]

            if prompt_style==1:
                prediction_prob = llm_sort_1(test_dataset=test_dataset, target_obj=target_obj)
                time.sleep(5)
            elif prompt_style==2:
                prediction_prob = llm_sort_2(test_dataset=test_dataset, target_obj=target_obj)
                time.sleep(5)
            elif prompt_style==3:
                prediction_prob = llm_sort_3(test_dataset=test_dataset, target_obj=target_obj)
            elif prompt_style==4:
                prediction_prob = llm_sort_4(test_dataset=test_dataset, target_obj=target_obj)
            elif prompt_style==5:
                prediction_prob = llm_sort_5(test_dataset=test_dataset, target_obj=target_obj)
            elif prompt_style==6:
                prediction_prob = llm_sort_6(test_dataset=test_dataset, target_obj=target_obj)
            elif prompt_style==7:
                prediction_prob = llm_sort_7(test_dataset=test_dataset, target_obj=target_obj)
            else:
                raise NameError("No such prompt")
            
            for j in range(len(dataset_list[i])):
                if te.columns_[target_key] not in test_dataset[j]:
                    dataset_list[i][j].append(prediction_prob[j])
                else:
                    dataset_list[i][j].append(2)
        print("prob prediction time: ", time.time()-start_time_1)


    elif optimization == "focusEZ":
        start_time_1 = time.time()
        if dataset_name == "Youtube-8M_seg":
            cluster_dir = "../../data/Youtube-8M_seg/index_cluster_youtube.json"
        else:
            cluster_dir = "../../data/HowTo100M/index_cluster_how.json"
        with open(cluster_dir, "r") as f:
            index_cluster = json.load(f)

        for i in range(len(key_candidates)):
            target_key = key_candidates[i]
            for cluster_id in index_cluster.keys():
                if index_cluster[cluster_id][1][target_key] == 1:
                    for j in index_cluster[cluster_id][0]:
                        dataset_list[i][j].append(1)
                else:
                    for j in index_cluster[cluster_id][0]:
                        dataset_list[i][j].append(0)
            for j in range(len(dataset_list[i])):
                if te.columns_[target_key] in test_dataset[j]:
                    dataset_list[i][j][-1] = 2
        print("prob prediction time: ", time.time()-start_time_1)

    elif optimization == 'kgEZ':
        numberbatch_model = gensim.models.KeyedVectors.load_word2vec_format('../../data/ConceptNet/numberbatch-en-19.08.txt', binary=False)
        page_views_avg = pd.read_pickle(dataset_name+"_pageview.pkl")
        page_views_prob = np.sqrt(page_views_avg/max(page_views_avg))

        start_time_1 = time.time()
        for i in range(len(key_candidates)):
            target_key = key_candidates[i]

            for j in range(len(dataset_list[i])):
                test_object = test_dataset[j]
                if te.columns_[target_key] in test_object:
                    dataset_list[i][j].append(2)
                    continue
                cond_object = []
                for obj in test_object:
                    if obj != te.columns_[target_key] and obj.replace(" ", "_") in numberbatch_model:
                        cond_object.append(obj)
                video_prob = cond_prob(cond_object=cond_object, target_obj=[te.columns_[target_key]], embeddings=numberbatch_model, single_prob_df=page_views_prob)
                dataset_list[i][j].append(min(video_prob, 1))
        print("prob prediction time: ", time.time()-start_time_1)

    elif optimization == "BertEZ":
        start_time_1 = time.time()
        for i in range(len(key_candidates)):
            target_key = key_candidates[i]
            prediction_prob = predict_prob_bert(test_dataset=test_dataset, te=te, target_key=target_key, model_dir="../model_prob_info/bert-base-uncased")
            for j in range(len(dataset_list[i])):
                if te.columns_[target_key] not in test_dataset[j]:
                    dataset_list[i][j].append(prediction_prob[j])
                else:
                    dataset_list[i][j].append(2)
        print("prob prediction time: ", time.time()-start_time_1)

    elif optimization == 'visualEZ': 
        start_time_1 = time.time()
        for i in range(len(key_candidates)):
            target_key = key_candidates[i]
            prediction_prob = predict_prob_bert(test_dataset=test_dataset, te=te, target_key=target_key, model_dir="../model_prob_info/"+checkpoint)
            for j in range(len(dataset_list[i])):
                if te.columns_[target_key] not in test_dataset[j]:
                    dataset_list[i][j].append(prediction_prob[j])
                else:
                    dataset_list[i][j].append(2)
        print("prob prediction time: ", time.time()-start_time_1)
    
    elif optimization == 'specialNNEZ':
        start_time_1 = time.time()

        specialNN_dir = "../../data/"+dataset_name+"/video_label/yolo-9000-clip-select-split/test_specialNN_9000/"
        predict_scores = []
        for j in range(len(test_labels)):
            video_clip_id = test_labels[j].split('.')[0]
            with open(specialNN_dir+video_clip_id+".txt", "rb") as f:
                predict_scores.append(pickle.load(f))

        for i in range(len(key_candidates)):
            target_key = key_candidates[i]

            for j in range(len(dataset_list[i])):
                dataset_list[i][j].append(max(predict_scores[j][te.columns_[target_key]]))
        print("prob prediction time: ", time.time()-start_time_1)

    elif optimization == 'randomEZ':
        start_time_1 = time.time()
        for i in range(len(key_candidates)):
            target_key = key_candidates[i]
            for j in range(len(dataset_list[i])):
                if te.columns_[target_key] not in test_dataset[j]:
                    dataset_list[i][j].append(random.random())
                else:
                    dataset_list[i][j].append(2)
        print("prob prediction time: ", time.time()-start_time_1)
    
    elif optimization == "diffEZ":
        def mse(img1, img2):
            err = np.sum((img1.astype("float") - img2.astype("float")) ** 2)
            err /= float(img1.shape[0] * img1.shape[1])
            return err
        
        diff_thresh = 1
        test_diff_dataset = []
        for i in range(len(test_labels)):
            test_label = test_labels[i]
            if test_label[:-4] not in os.listdir(test_true_frames_dir):
                test_diff_dataset.append(test_dataset[i])
            else:
                test_df = pd.read_csv(test_dir+test_label)
                test_true_df = pd.read_csv(test_true_dir+test_label)
                test_len = len(test_df["objects"])
                test_true_len = len(test_true_df["objects"])
                frame_rate = int((test_true_len-2)/(test_len-1))

                satisfy_diff = True # determine whether the current index (1 frame per second) satisfies difference threshold.
                pre_image = cv2.imread(test_true_frames_dir+test_label[:-4]+"/frame0.jpg")
                for j in range(1, test_len):
                    if "frame"+str(j*frame_rate)+".jpg" in os.listdir(test_true_frames_dir+test_label[:-4]):
                        cur_image = cv2.imread(test_true_frames_dir+test_label[:-4]+"/frame"+str(j*frame_rate)+".jpg")
                    else:
                        break
                    if mse(pre_image, cur_image) <= diff_thresh:
                        satisfy_diff = False
                    pre_image = cur_image

                if satisfy_diff == True:
                    test_diff_dataset.append(test_dataset[i])
                else: # if the index does not satisfy, sample from original frames to satisfy the difference threshold.
                    frame_id = [0]
                    pre_image = cv2.imread(test_true_frames_dir+test_label[:-4]+"/frame0.jpg")
                    for j in range(1, test_true_len-1):
                        if "frame"+str(j)+".jpg" in os.listdir(test_true_frames_dir+test_label[:-4]):
                            cur_image = cv2.imread(test_true_frames_dir+test_label[:-4]+"/frame"+str(j)+".jpg")
                        else:
                            break
                        if mse(pre_image, cur_image) > diff_thresh:
                            frame_id.append(j)
                            pre_image = cur_image
                    if len(frame_id) > test_len:
                        frame_id = random.sample(frame_id, test_len)
                    else:
                        frame_id += random.sample(range(test_true_len-1), test_len-len(frame_id))
                    
                    diff_object = []
                    for fid in frame_id:
                        if type(test_true_df["objects"][fid+1]) != np.float64 and type(test_true_df["objects"][fid+1]) != float:
                            obj_res = test_true_df["objects"][fid+1].split(',')
                            for obj in obj_res:
                                if obj.split(':')[0] not in diff_object and obj.split(':')[0] in te.columns_:
                                    diff_object.append(obj.split(':')[0])
                    test_diff_dataset.append(diff_object)

        start_time_1 = time.time()
        for i in range(len(key_candidates)):
            target_key = key_candidates[i]
            for j in range(len(dataset_list[i])):
                if te.columns_[target_key] not in test_diff_dataset[j]:
                    dataset_list[i][j].append(random.random())
                else:
                    dataset_list[i][j].append(2)
        print("prob prediction time: ", time.time()-start_time_1)

    else:
        raise NameError('Wrong optimization name')

    return dataset_list

def process(dataset_list):
    '''
    Process the videos according to the order of pred_prob.
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
            model_process_num = 0
            while res_size < limit:
                next_item = cur_labels[process_idx]
                if next_item[0] == 1:
                    res_size += 1
                process_idx += 1
                if next_item[1] < 2:
                    model_process_num += 1
            process_num.append(model_process_num)
        process_num_list.append(process_num)

    print("time of selecting video candidates for all the LIMIT settings: ", time.time()-start_time_2)
    
    return process_num_list

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--all_candidates', required=True)
    parser.add_argument('--key_candidates')
    parser.add_argument('--optimization', required=True)
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--repeat', type=int, required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--index_percent', type=int, required=True) # 100
    parser.add_argument('--result_name', required=True) # _finetune_all
    parser.add_argument('--prompt_style', type=int)
    args = parser.parse_args()

    all_candidates = args.all_candidates
    optimization = args.optimization
    dataset_name = args.dataset
    repeat_times = args.repeat
    checkpoint = args.checkpoint
    index_percent = args.index_percent
    result_name = args.result_name
    if all_candidates == 'True':
        if dataset_name == "Youtube-8M_seg":
            with open("../../BERT_result/raw_result/Youtube-8M_seg/selected_keys_10_unvague_Youtube.txt", 'rb') as f:
                key_candidates = pickle.load(f)
                # key_candidates = key_candidates[:10]
        elif dataset_name == "HowTo100M":
            with open("../../BERT_result/raw_result/HowTo100M/selected_keys_10_unvague_How.txt", 'rb') as f:
                key_candidates = pickle.load(f)
        else:
            raise NameError('Wrong Dataset Name')
    else:
        key_candidates = args.key_candidates
        key_candidates = [int(item) for item in key_candidates.split(',')]
    prompt_style = 0
    if optimization == "llmEZ":
        prompt_style = args.prompt_style

    train_dir = "../../data/"+dataset_name+"/video_label/yolo-9000-clip-select-split/train/"
    val_dir = "../../data/"+dataset_name+"/video_label/yolo-9000-clip-select-split/val/"
    test_dir = "../../data/"+dataset_name+"/video_label/yolo-9000-clip-select-split/test/"
    test_true_dir = "../../data/"+dataset_name+"/video_label/yolo-9000-clip-select-split/test_true/"
    test_true_frames_dir = "/z/wenjiah/query_commonsense/data/"+dataset_name+"/video_label/yolo-9000-clip-select-split/test_true_frames/"
    if dataset_name == "Youtube-8M_seg":
        finetune_filename = "finetune_videos_youtube"
    elif dataset_name == "HowTo100M":
        finetune_filename = "finetune_videos_How"
    else:
        raise NameError('Wrong Dataset Name')

    train_dataset, test_dataset, test_true_dataset, te, dataset_list, test_labels = construct_dataset(train_dir, test_dir, test_true_dir, key_candidates, finetune_filename, index_percent)
    start_time_total = time.time()
    process_num_avg_list = np.zeros([len(key_candidates), 10])
    for trial in range(repeat_times):
        predicted_dataset_list = video_pred_prob(optimization, test_dataset, te, copy.deepcopy(dataset_list), key_candidates, dataset_name, test_labels, test_dir, test_true_dir, test_true_frames_dir, checkpoint, prompt_style)
        process_num_list = process(predicted_dataset_list)
        process_num_avg_list += np.array(process_num_list)
    process_num_avg_list /= repeat_times     
    # print("Total time of 10 processing with ", optimization, ": ", time.time()-start_time_total)   

    store_dict = {}
    store_dict["process num"] = process_num_avg_list
    store_dict["keys"] = key_candidates
    key_names = []
    for i in range(len(key_candidates)):
        key_names.append(te.columns_[key_candidates[i]])
    store_dict["key names"] = key_names

    if dataset_name == "Youtube-8M_seg":
        suffix = "Youtube"
    elif dataset_name == "HowTo100M":
        suffix = "How"
    else:
        raise NameError('Wrong Dataset Name')
    with open("../../BERT_result/raw_result/"+dataset_name+"/opt_"+suffix+"_"+optimization+result_name+".txt", 'wb') as f:
    # with open("../../BERT_result/raw_result/"+dataset_name+"/opt_"+suffix+"_"+optimization+"_diff1_finetune_all.txt", 'wb') as f:
        pickle.dump(store_dict, f)
