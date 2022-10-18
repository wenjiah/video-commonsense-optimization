import os
import pandas as pd
from pytube import YouTube

df = pd.read_csv("videoid_convert.csv")

for vid in df['youtubeid']:
    url = "https://www.youtube.com/watch?v="+vid
    
    try:  
        yt = YouTube(url)  
    except:  
        print("Connection Error") 
    
    # filters out the files with "mp4" extension
    try:  
        mp4file = yt.streams.filter(file_extension = "mp4").first() 
    except:
        print("Maybe a private video") 

    try:  
        mp4file.download(output_path="video/", filename=vid+".mp4")  
    except:  
        print("Some Error!")  

print("Download all videos!")