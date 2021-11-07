import os
import json
import pandas as pd
import requests
import shutil
from pytube import YouTube

df = pd.read_csv("videoid_convert.csv")

for vid in df['youtubeid']:
    url = "https://www.youtube.com/watch?v="+vid
    
    try:  
        yt = YouTube(url)  
    except:  
        print("Connection Error") 
    
    # filters out the files with "mp4" extension  
    mp4file = yt.streams.filter(file_extension = "mp4").first()  

    try:  
        mp4file.download(output_path="./video/", filename=vid)  
    except:  
        print("Some Error!")  