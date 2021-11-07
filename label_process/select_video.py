import pandas as pd
import os
import numpy as np

'''
YouTube-8M segment dataset
'''
# video_label_dir = "/z/wenjiah/query_commonsense/data/Youtube-8M_seg/video_label/yolo-9000-clip/"
# copy_dir = "/z/wenjiah/query_commonsense/data/Youtube-8M_seg/video_label/yolo-9000-clip-select/"

# for video_label in os.listdir(video_label_dir):
#     label_path = video_label_dir + video_label
#     df = pd.read_csv(label_path)

#     detected_object = set()
#     for i in range(len(df["objects"])):
#         if type(df["objects"][i]) != np.float64 and type(df["objects"][i]) != float:
#             obj_res = df["objects"][i].split(',')
#             for obj in obj_res:
#                 if obj.split(':')[0] not in detected_object:
#                     detected_object.add(obj.split(':')[0])

#     if len(detected_object) >= 5:
#         os.system("cp " + label_path + " " + copy_dir + video_label)

'''
HowTo100M dataset
'''
video_label_dir = "/z/wenjiah/query_commonsense/data/HowTo100M/video_label/yolo-9000-clip/"
copy_dir = "/z/wenjiah/query_commonsense/data/HowTo100M/video_label/yolo-9000-clip-select/"

for video_label in os.listdir(video_label_dir):
    label_path = video_label_dir + video_label
    df = pd.read_csv(label_path)

    detected_object = set()
    for i in range(len(df["objects"])):
        if type(df["objects"][i]) != np.float64 and type(df["objects"][i]) != float:
            obj_res = df["objects"][i].split(',')
            for obj in obj_res:
                if obj.split(':')[0] not in detected_object:
                    detected_object.add(obj.split(':')[0])

    if len(detected_object) >= 5:
        os.system("cp " + label_path + " " + copy_dir + video_label)
