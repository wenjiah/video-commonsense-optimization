import subprocess
import pandas as pd
import os
import cv2

cmd = ['./darknet', 'detector', 'test', 'cfg/combine9k.data', 'cfg/yolo9000.cfg', '../yolo9000-weights/yolo9000.weights', '-gpus', '0']

video_dir = "/z/wenjiah/query_commonsense/data/Youtube-8M_seg/video"
# frame_dir = "/z/wenjiah/query_commonsense/data/Youtube-8M_seg/part_video_label/yolo-9000-clip-select-split/test_true_frames"
# test_dir = "/z/wenjiah/query_commonsense/data/Youtube-8M_seg/part_video_label/yolo-9000-clip-select-split/test"
frame_dir = "/z/wenjiah/query_commonsense/data/Youtube-8M_seg/video_label/yolo-9000-clip-select-split/test_true_frames"
test_dir = "/z/wenjiah/query_commonsense/data/Youtube-8M_seg/video_label/yolo-9000-clip-select-split/test"
total_label_files = os.listdir(test_dir)

for label_file in total_label_files[:int(0.35*len(total_label_files))]:
    video_id = label_file[:-6]
    time_range = (60*int(label_file[-5]), 60*(int(label_file[-5])+1))
    video_path = video_dir + "/" + video_id + ".mp4"
    frame_path = frame_dir + "/" + label_file.split('.')[0]
    os.mkdir(frame_path)

    vidcap = cv2.VideoCapture(video_path)
    vidcap.set(cv2.CAP_PROP_POS_MSEC,(time_range[0]*1000))  
    success,image = vidcap.read()
    count = 0
    while success and vidcap.get(cv2.CAP_PROP_POS_MSEC) < time_range[1]*1000:
        cv2.imwrite(frame_path + "/" + "frame%d.jpg"%count, image)   
        count = count + 1
        success,image = vidcap.read()

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
    # df.to_csv("/z/wenjiah/query_commonsense/data/Youtube-8M_seg/part_video_label/yolo-9000-clip-select-split/test_true/" + label_file, index=None)
    df.to_csv("/z/wenjiah/query_commonsense/data/Youtube-8M_seg/video_label/yolo-9000-clip-select-split/test_true/" + label_file, index=None)

print("All Done!")