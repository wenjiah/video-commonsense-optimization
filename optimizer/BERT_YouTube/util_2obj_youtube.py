import numpy as np
from random import sample

def CustomDataset(dataset, te, target_key):
    custom_dataset = {}
    text_list = []
    label_list = []            
    
    for video_objects in dataset:
        text = ""
        for obj in video_objects:
            if obj != te.columns_[target_key[0]] and obj != te.columns_[target_key[1]]:
                text += obj + " "
        text_list.append([text, te.columns_[target_key[0]]+" "+te.columns_[target_key[1]]])

        if te.columns_[target_key[0]] in video_objects and te.columns_[target_key[1]] in video_objects:
            label_list.append(float(1))
        else:
            label_list.append(float(0))

    custom_dataset["text"] = text_list
    custom_dataset["label"] = label_list
    return custom_dataset

