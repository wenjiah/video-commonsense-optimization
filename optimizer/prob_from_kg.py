import time
import pandas as pd
import os
import numpy as np
import random
import copy
import pickle


def single_prob(single_prob_df, obj_name):
    obj_name = obj_name.capitalize().replace(" ", "_")

    if single_prob_df[obj_name] > 0:
        est_single_prob = single_prob_df[obj_name]
    else:
        est_single_prob = 0.0001

    return est_single_prob

def two_prob(obj_list, single_prob_df, embeddings):
    jaccard_prob = max(0.01, embeddings.similarity(obj_list[0].replace(" ", "_"), obj_list[1].replace(" ", "_")))
    est_two_prob = ((single_prob(single_prob_df, obj_list[0]) + single_prob(single_prob_df, obj_list[1])) * jaccard_prob) / (1 + jaccard_prob)

    return est_two_prob

def multi_prob(obj_list, single_prob_df, embeddings):
    pair_prob = []

    for i in range(len(obj_list)-1):
        for j in range(i+1, len(obj_list)):
            pair_prob.append(two_prob([obj_list[i],obj_list[j]], single_prob_df, embeddings))

    if len(obj_list)%2 != 0:
        for i in range(len(obj_list)):
            pair_prob.append(single_prob(single_prob_df, obj_list[i]))

    multi_prob_ub = min(pair_prob)

    if len(obj_list)%2 == 0:
        multi_prob_lb = max(0, sum(pair_prob)/(len(obj_list)-1)-(len(obj_list)/2-1))
    else:
        multi_prob_lb = max(0, sum(pair_prob)/len(obj_list)-((len(obj_list)+1)/2-1))

    return (multi_prob_lb+multi_prob_ub)/2


def cond_prob(cond_object, target_obj, embeddings, single_prob_df):
    if len(cond_object) == 0:
        est_cond_prob = 0

    elif len(cond_object) == 1:
        if len(target_obj) == 1:
            est_cond_prob = two_prob(cond_object+target_obj, single_prob_df, embeddings) / single_prob(single_prob_df, cond_object[0])
        else:
            if len(target_obj) == 2:
                est_cond_prob = two_prob(target_obj, single_prob_df, embeddings)
            else:
                est_cond_prob = multi_prob(target_obj, single_prob_df, embeddings)
            est_cond_prob = est_cond_prob * cond_prob(cond_object=target_obj, target_obj=cond_object, embeddings=embeddings, single_prob_df=single_prob_df) / single_prob(single_prob_df, cond_object[0])
    
    elif len(cond_object) == 2:
        if two_prob(cond_object, single_prob_df, embeddings) == 0:
            est_cond_prob = 0
        else:
            if len(target_obj) == 1:
                est_cond_prob = single_prob(single_prob_df, target_obj[0])
            elif len(target_obj) == 2:
                est_cond_prob = two_prob(target_obj, single_prob_df, embeddings)
            else:
                est_cond_prob = multi_prob(target_obj, single_prob_df, embeddings)
            for i in range(len(cond_object)):
                est_cond_prob *= cond_prob(cond_object=target_obj, target_obj=cond_object[i:i+1], embeddings=embeddings, single_prob_df=single_prob_df)
            est_cond_prob /= two_prob(cond_object, single_prob_df, embeddings)
    
    else:
        if multi_prob(cond_object, single_prob_df, embeddings) == 0:
            est_cond_prob = 0
        else:
            if len(target_obj) == 1:
                est_cond_prob = single_prob(single_prob_df, target_obj[0])
            elif len(target_obj) == 2:
                est_cond_prob = two_prob(target_obj, single_prob_df, embeddings)
            else:
                est_cond_prob = multi_prob(target_obj, single_prob_df, embeddings)
            for i in range(len(cond_object)):
                est_cond_prob *= cond_prob(cond_object=target_obj, target_obj=cond_object[i:i+1], embeddings=embeddings, single_prob_df=single_prob_df)
            est_cond_prob /= multi_prob(cond_object, single_prob_df, embeddings)

    return est_cond_prob