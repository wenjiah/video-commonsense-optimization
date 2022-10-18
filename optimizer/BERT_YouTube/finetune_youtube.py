import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
import os
import numpy as np
import random
import pickle
import time
from util_youtube import CustomDataset, ComputeMetric  
from datasets import Dataset 
# import wandb     
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer, DataCollatorWithPadding

os.environ["WANDB_DISABLED"] = "true"

train_dataset = [] 
train_dir = "../../data/Youtube-8M_seg/video_label/yolo-9000-clip-select-split/train/"
for train_label in os.listdir(train_dir):
    label_path = train_dir + train_label
    df = pd.read_csv(label_path)

    detected_object = []
    for i in range(len(df["objects"])):
        if type(df["objects"][i]) != np.float64 and type(df["objects"][i]) != float:
            obj_res = df["objects"][i].split(',')
            for obj in obj_res:
                if obj.split(':')[0] not in detected_object:
                    detected_object.append(obj.split(':')[0])
    train_dataset.append(detected_object)

te = TransactionEncoder()
te = te.fit(train_dataset)

finetune_videos = []
test_true_dataset = []
test_true_dir = "../../data/Youtube-8M_seg/video_label/yolo-9000-clip-select-split/test_true/"
video_interval = 2
label_count = 0
for test_true_label in os.listdir(test_true_dir):
    label_count += 1
    if label_count%video_interval == 0:
        finetune_videos.append(test_true_label)
        label_path = test_true_dir + test_true_label
        df = pd.read_csv(label_path)

        test_true_object = []
        for i in range(len(df["objects"])):
            if type(df["objects"][i]) != np.float64 and type(df["objects"][i]) != float:
                obj_res = df["objects"][i].split(',')
                for obj in obj_res:
                    if obj.split(':')[0] not in test_true_object and obj.split(':')[0] in te.columns_:
                        test_true_object.append(obj.split(':')[0])
        test_true_dataset.append(test_true_object)
random.seed(30)
random.shuffle(test_true_dataset)
with open("../../model/finetune_videos_youtube.txt", 'wb') as f:
    pickle.dump(finetune_videos, f)

custom_dataset_train = Dataset.from_dict(CustomDataset(dataset=test_true_dataset[:int(0.8*len(test_true_dataset))], te=te, target_key=-1))

custom_dataset_val = {"text":[], "label":[]}
with open("../../result/Youtube-8M_seg/selected_keys_10_unvague_Youtube.txt", 'rb') as f:
    key_candidates = pickle.load(f)
for target_key in key_candidates:
    val_group = CustomDataset(dataset=test_true_dataset[int(0.8*len(test_true_dataset)):], te=te, target_key=target_key)
    custom_dataset_val["text"] += val_group["text"]
    custom_dataset_val["label"] += val_group["label"]
custom_dataset_val = Dataset.from_dict(custom_dataset_val)

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

def tokenize_function(examples):
    return tokenizer(examples["text"], truncation=True, max_length=512)

training_data = custom_dataset_train.map(tokenize_function, batched=True, load_from_cache_file=False)
val_data = custom_dataset_val.map(tokenize_function, batched=True, load_from_cache_file=False)

model = AutoModelForSequenceClassification.from_pretrained("../../model/train_youtube/checkpoint-4000")
training_args = TrainingArguments(output_dir="../../model/finetune_youtube", evaluation_strategy="steps", warmup_ratio=0.1, logging_steps=500, per_device_train_batch_size=64, per_device_eval_batch_size=512, num_train_epochs=2, learning_rate=2e-05, load_best_model_at_end = True, metric_for_best_model = 'avg processed videos', greater_is_better = False, save_total_limit=1)

trainer = Trainer(
    model = model,
    args = training_args,
    train_dataset = training_data,
    eval_dataset = val_data,
    compute_metrics = ComputeMetric,
    data_collator=DataCollatorWithPadding(tokenizer=tokenizer)
)

start = time.time()
trainer.train()
print("Finetune training time is ", time.time()-start)