import os
import json
import pandas as pd
from pandas.core.indexes import category
import requests
import shutil
import pytube
from pytube import YouTube

category_names = ["Arts and Entertainment", "Cars & Other Vehicles", "Computers and Electronics", "Education and Communications", "Food and Entertaining", "Health", "Hobbies and Crafts", "Holidays and Traditions", "Home and Garden", "Personal Care and Style", "Pets and Animals", "Sports and Fitness"]

df = pd.read_csv("HowTo100M/HowTo100M_v1.csv")
sample_df = None
for name in category_names:
    if sample_df is None:
        sample_df = df[df["category_1"] == name].sample(n=5000)
    else:
        sample_df = pd.concat([sample_df, df[df["category_1"] == name].sample(n=5000)])

sample_df.to_csv("video.csv", index=None)

for video_id in sample_df["video_id"]:
    url = "https://www.youtube.com/watch?v="+video_id

    try:  
        yt = YouTube(url)  
    except:  
        print("Connection Error") 
        continue

    # filters out the files with "mp4" extension  
    try:
        mp4file = yt.streams.filter(file_extension = "mp4").first()  
    except pytube.exceptions.VideoUnavailable:
        print("Unavailable")
        continue
    except KeyError:
        print("KeyError")
        continue
    
    try:  
        mp4file.download(output_path="./video/", filename=video_id+".mp4")  
    except:  
        print("Some Error!")  
        continue

