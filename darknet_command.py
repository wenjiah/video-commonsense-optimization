import subprocess
import time
import io
import pandas as pd
import os
import cv2

cmd = ['./darknet', 'detector', 'test', 'cfg/combine9k.data', 'cfg/yolo9000.cfg', '../yolo9000-weights/yolo9000.weights']

video_dir = "../data/Youtube-8M_seg/video"
frame_dir = "../data/Youtube-8M_seg/video_frame"
# video_dir = "../data/HowTo100M/video"
# frame_dir = "../data/HowTo100M/video_frame"

for video_name in os.listdir(video_dir):
    video_path = video_dir + "/" + video_name
    frame_path = frame_dir + "/" + video_name.split('.')[0]
    os.mkdir(frame_path)

    count = 0
    vidcap = cv2.VideoCapture(video_path)
    success,image = vidcap.read()
    success = True
    while success and count<500:
        vidcap.set(cv2.CAP_PROP_POS_MSEC,(count*1000))    # added this line 
        success,image = vidcap.read()
        if success:
            cv2.imwrite(frame_path + "/" + "frame%d.jpg"%count, image)     # save frame as JPEG file
            count = count + 1

    p = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr = subprocess.PIPE)

    inputs = ''
    for i in range(count):
        inputs += frame_path + "/" + "frame%d.jpg\n"%i
    results = p.communicate(inputs.encode('utf-8'))[0]
    results = results.decode('utf-8').split('\n')

    res = []
    frame_idx = 0
    frame_res = ""
    for i in range(1, len(results)):
        if results[i].startswith("Enter Image Path"):
            if frame_res == "":
                res.append([frame_idx, frame_res])
            else:
                res.append([frame_idx, frame_res[:-1]])
            frame_idx += 1
            frame_res = ""
        else:
            frame_res += results[i]+','

    df = pd.DataFrame(res, columns=['frame', 'objects'])
    df.to_csv("../data/Youtube-8M_seg/video_label/yolo-9000/" + video_name.split('.')[0] + ".csv", index=None)
    # df.to_csv("../data/HowTo100M/video_label/yolo-9000/" + video_name.split('.')[0] + ".csv", index=None)

print("All Done!")