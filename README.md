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

Copy *darknet_command.py* and *darknet_command_true.py* to *./yolo-9000/darknet*, and process video frames with the detector YOLO9000 to collect object information.
```
cd ./yolo-9000/darknet
python darknet_command.py
python darknet_command_true.py
```

## Preprocess video with object information
Cut long videos into 60-second video clips, select video clips with at least five distinct objects, and split the selected corpus into three sets for model training, validation, and testing.
```
cd ./label_process
python video_clips.py
python select_video.py
python split_video.py
```

## Video selection query optimization
Select target objects
```
cd ./optimizer
python select_obj.py
python selecy_2obj.py
```

Train the BERT regression model on YouTube-8M Segments data and apply the online learning strategy. (The process would be similar for HowTo100M data.)
```
cd ./optimizer/BERT_YouTube
python train_youtube.py
python finetune_youtube.py
```

Optimize video selection queries by the model built with videos
```
cd ./optimizer
python optimizer.py --all_candidates True --optimization visualEZ --dataset Youtube-8M_seg --repeat 1 --checkpoint finetune_youtube/checkpoint-1500 --index_percent 100 --result_name _finetune_all
python optimizer_2obj.py --all_candidates True --optimization visualEZ --dataset Youtube-8M_seg --repeat 1 --checkpoint finetune_youtube/checkpoint-1500 --result_name _finetune_2obj_all
```

Collect pageview statistics of Wikipedia
```
cd ./optimizer
python obj_pageview.py
```

Optimize video selection queries by the model built without videos
```
cd ./optimizer
python optimizer.py --all_candidates True --optimization kgEZ --dataset Youtube-8M_seg --repeat 1 --checkpoint finetune_youtube/checkpoint-1500 --index_percent 100 --result_name _finetune_all
python optimizer_2obj.py --all_candidates True --optimization kgEZ --dataset Youtube-8M_seg --repeat 1 --checkpoint finetune_youtube/checkpoint-1500 --result_name _finetune_2obj_all
```

Optimize video selection queries by other baselines
```
cd ./optimizer
python optimizer.py --all_candidates True --optimization focusEZ --dataset Youtube-8M_seg --repeat 1 --checkpoint finetune_youtube/checkpoint-1500 --index_percent 100 --result_name _finetune_all
python optimizer.py --all_candidates True --optimization diffEZ --dataset Youtube-8M_seg --repeat 1 --checkpoint finetune_youtube/checkpoint-1500 --index_percent 100 --result_name _finetune_all
python optimizer.py --all_candidates True --optimization randomEZ --dataset Youtube-8M_seg --repeat 1 --checkpoint finetune_youtube/checkpoint-1500 --index_percent 100 --result_name _finetune_all
```

Use pre-trained BERT model to predict the existence probability of target objects
```
cd ./optimizer
python optimizer.py --all_candidates True --optimization BertEZ --dataset Youtube-8M_seg --repeat 1 --checkpoint finetune_youtube/checkpoint-1500 --index_percent 100 --result_name _finetune_all
```

Use LLM(GPT-3.5) to rank videos so as to optimize video selection queries
```
cd ./optimizer
python optimizer.py --all_candidates True --optimization llmEZ --dataset Youtube-8M_seg --repeat 1 --checkpoint finetune_youtube/checkpoint-1500 --index_percent 100 --result_name _2_finetune_all --prompt_style 2
```
See the ten tested prompts in *./optimizer/GPT_prompts.pdf*
