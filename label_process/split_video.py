import os
import numpy as np

'''
YouTube-8M segment dataset
'''
# select_video_dir = "/z/wenjiah/query_commonsense/data/Youtube-8M_seg/video_label/yolo-9000-clip-select/"
# split_video_dir = "/z/wenjiah/query_commonsense/data/Youtube-8M_seg/video_label/yolo-9000-clip-select-split/"

# total_labels = os.listdir(select_video_dir)
# video_num = len(total_labels)
# permutation = np.random.permutation(video_num)

# video_index = {}
# for i in range(video_num):
#     video_name = total_labels[i][:-6]
#     if video_name not in video_index.keys():
#         video_index[video_name] = [total_labels[i]]
#     else:
#         video_index[video_name].append(total_labels[i])

# added_video = set()
# train_clips = []
# val_clips = []
# test_clips = []

# for i in range(video_num):
#     if total_labels[permutation[i]][:-6] not in added_video:
#         added_video.add(total_labels[permutation[i]][:-6])
#         if len(train_clips) < int(0.6*video_num):
#             train_clips += video_index[total_labels[permutation[i]][:-6]]
#         elif len(val_clips) < int(0.2*video_num):
#             val_clips += video_index[total_labels[permutation[i]][:-6]]
#         else:
#             test_clips += video_index[total_labels[permutation[i]][:-6]]

# for filename in train_clips:
#     os.system("cp " + select_video_dir + filename + " " + split_video_dir + "train/" + filename)
# for filename in val_clips:
#     os.system("cp " + select_video_dir + filename + " " + split_video_dir + "val/" + filename)
# for filename in test_clips:
#     os.system("cp " + select_video_dir + filename + " " + split_video_dir + "test/" + filename)

'''
HowTo100M dataset
'''
select_video_dir = "/z/wenjiah/query_commonsense/data/HowTo100M/video_label/yolo-9000-clip-select/"
split_video_dir = "/z/wenjiah/query_commonsense/data/HowTo100M/video_label/yolo-9000-clip-select-split/"

total_labels = os.listdir(select_video_dir)
video_num = len(total_labels)
permutation = np.random.permutation(video_num)

video_index = {}
for i in range(video_num):
    video_name = total_labels[i][:-6]
    if video_name not in video_index.keys():
        video_index[video_name] = [total_labels[i]]
    else:
        video_index[video_name].append(total_labels[i])

added_video = set()
train_clips = []
val_clips = []
test_clips = []

for i in range(video_num):
    if total_labels[permutation[i]][:-6] not in added_video:
        added_video.add(total_labels[permutation[i]][:-6])
        if len(train_clips) < int(0.6*video_num):
            train_clips += video_index[total_labels[permutation[i]][:-6]]
        elif len(val_clips) < int(0.2*video_num):
            val_clips += video_index[total_labels[permutation[i]][:-6]]
        else:
            test_clips += video_index[total_labels[permutation[i]][:-6]]

for filename in train_clips:
    os.system("cp " + select_video_dir + filename + " " + split_video_dir + "train/" + filename)
for filename in val_clips:
    os.system("cp " + select_video_dir + filename + " " + split_video_dir + "val/" + filename)
for filename in test_clips:
    os.system("cp " + select_video_dir + filename + " " + split_video_dir + "test/" + filename)
