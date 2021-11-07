import pandas as pd
import os
import numpy as np

'''
YouTube-8M segment dataset
'''
# all_detected_object = []
# all_detected_object_num = []
# total_category = set()
# video_label_dist = [0 for i in range(8)] # 0, 1, 2, ..., 6, >=7
# frame_label_dist = [0 for i in range(8)]

# video_label_dir = "/z/wenjiah/query_commonsense/data/Youtube-8M_seg/video_label/yolo-9000-clip-select/"

# for video_label in os.listdir(video_label_dir):
#     label_path = video_label_dir + video_label
#     df = pd.read_csv(label_path)

#     detected_object = set()
#     for i in range(len(df["objects"])):
#         frame_detected_object = set()
#         if type(df["objects"][i]) != np.float64 and type(df["objects"][i]) != float:
#             obj_res = df["objects"][i].split(',')
#             for obj in obj_res:
#                 if obj.split(':')[0] not in detected_object:
#                     detected_object.add(obj.split(':')[0])
#                 if obj.split(':')[0] not in frame_detected_object:
#                     frame_detected_object.add(obj.split(':')[0])
#                 if obj.split(':')[0] not in total_category:
#                     total_category.add(obj.split(':')[0])

#         if len(frame_detected_object) < 7:
#             frame_label_dist[len(frame_detected_object)] += 1
#         else:
#             frame_label_dist[7] += 1

#     all_detected_object.append(detected_object)
#     all_detected_object_num.append(len(detected_object))

#     if len(detected_object) < 7:
#         video_label_dist[len(detected_object)] += 1
#     else:
#         video_label_dist[7] += 1

# print("video level: ", video_label_dist)
# print("frame level: ", frame_label_dist)
# print("total categories: ", len(total_category))

'''
HowTo100M dataset
'''
all_detected_object = []
all_detected_object_num = []
total_category = set()
video_label_dist = [0 for i in range(8)] # 0, 1, 2, ..., 6, >=7
frame_label_dist = [0 for i in range(8)]

video_label_dir = "/z/wenjiah/query_commonsense/data/HowTo100M/video_label/yolo-9000-clip/"

for video_label in os.listdir(video_label_dir):
    label_path = video_label_dir + video_label
    df = pd.read_csv(label_path)

    detected_object = set()
    for i in range(len(df["objects"])):
        frame_detected_object = set()
        if type(df["objects"][i]) != np.float64 and type(df["objects"][i]) != float:
            obj_res = df["objects"][i].split(',')
            for obj in obj_res:
                if obj.split(':')[0] not in detected_object:
                    detected_object.add(obj.split(':')[0])
                if obj.split(':')[0] not in frame_detected_object:
                    frame_detected_object.add(obj.split(':')[0])
                if obj.split(':')[0] not in total_category:
                    total_category.add(obj.split(':')[0])

        if len(frame_detected_object) < 7:
            frame_label_dist[len(frame_detected_object)] += 1
        else:
            frame_label_dist[7] += 1

    all_detected_object.append(detected_object)
    all_detected_object_num.append(len(detected_object))

    if len(detected_object) < 7:
        video_label_dist[len(detected_object)] += 1
    else:
        video_label_dist[7] += 1

print("video level: ", video_label_dist)
print("frame level: ", frame_label_dist)
print("total categories: ", len(total_category))