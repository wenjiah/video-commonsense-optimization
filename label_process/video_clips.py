import pandas as pd
import os

'''
YouTube-8M segment dataset
'''
# video_label_dir = "/z/wenjiah/query_commonsense/data/Youtube-8M_seg/video_label/yolo-9000/"
# video_clip_dir = "/z/wenjiah/query_commonsense/data/Youtube-8M_seg/video_label/yolo-9000-clip/"
# clip_len = 60

# for video_label in os.listdir(video_label_dir):
#     label_path = video_label_dir + video_label
#     df = pd.read_csv(label_path) 

#     clip_num = len(df) // clip_len
#     for clip in range(clip_num):
#         clip_df = df[clip*clip_len:(clip+1)*clip_len]
#         clip_df.to_csv(video_clip_dir + video_label.split(".")[0] + "_%d"%clip + ".csv", index=None)

'''
HowTo100M dataset
'''
video_label_dir = "/z/wenjiah/query_commonsense/data/HowTo100M/video_label/yolo-9000/"
video_clip_dir = "/z/wenjiah/query_commonsense/data/HowTo100M/video_label/yolo-9000-clip/"
clip_len = 60

for video_label in os.listdir(video_label_dir):
    label_path = video_label_dir + video_label
    df = pd.read_csv(label_path) 

    clip_num = len(df) // clip_len
    for clip in range(clip_num):
        clip_df = df[clip*clip_len:(clip+1)*clip_len]
        clip_df.to_csv(video_clip_dir + video_label.split(".")[0] + "_%d"%clip + ".csv", index=None)