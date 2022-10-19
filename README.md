# video-commonsense-optimization

## Download data
Download the video data [*YouTube-8M Segments*](https://research.google.com/youtube8m/download.html)
```
cd ./data/Youtube-8M_seg
curl data.yt8m.org/download.py | partition=3/frame/validate mirror=us python
python videoid_convert.py
python download_video.py
```

Download the video data [*HowTo100M*](https://www.di.ens.fr/willow/research/howto100m/) (All-in-One zip) to *./data/HowTo100M*
```
cd ./data/HowTo100M
python download_video.py
```

Download [*ConceptNet Numberbatch*](https://github.com/commonsense/conceptnet-numberbatch) (numberbatch-en-19.08.txt) to *./data/ConceptNet*

## Object detection
Set up [YOLO9000](https://github.com/philipperemy/yolo-9000) with GPU support according to the instructions in README.md

Process video frames with the detector YOLO9000 to collect object information.

## Preprocess video data
Split large video into video clips and split the large video corpus into three different corpus for conditional relative frequency computation, model training, and testing. 

## Optimization
Optimize video queries with different methods.
