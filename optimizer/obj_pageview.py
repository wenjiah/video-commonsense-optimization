import mwviews
import pandas as pd
import numpy as np
from mwviews.api import PageviewsClient
import os
from itertools import chain

for dataset_name in ["Youtube-8M_seg", "HowTo100M"]:
    train_dir = "../data/"+dataset_name+"/video_label/yolo-9000-clip-select-split/train/"

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

    flatten_train_dataset = list(chain.from_iterable(train_dataset))
    obj_list = list(dict.fromkeys(flatten_train_dataset))
    for i in range(len(obj_list)):
        obj_list[i] = obj_list[i].capitalize()

    p = PageviewsClient(user_agent="Python query script")
    page_views = p.article_views(project='en.wikipedia', articles=obj_list, granularity='daily', start='2022010100', end='2022123123')
    page_views_df = pd.DataFrame(page_views)
    page_views_df_trans = page_views_df.transpose()
    page_views_avg = page_views_df_trans.sum() / len(page_views_df.iloc[0])

    print(page_views_avg)
    page_views_avg.to_pickle(dataset_name+"_pageview.pkl")