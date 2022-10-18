import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
import os
import numpy as np
import time
import random
import pickle
from util_youtube import CustomDataset, ComputeMetric
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer, DataCollatorWithPadding
# import wandb
from datasets import Dataset

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

val_dataset = []
val_dir = "../../data/Youtube-8M_seg/video_label/yolo-9000-clip-select-split/val/"
for val_label in os.listdir(val_dir):
    label_path = val_dir + val_label
    df = pd.read_csv(label_path)

    detected_object = []
    for i in range(len(df["objects"])):
        if type(df["objects"][i]) != np.float64 and type(df["objects"][i]) != float:
            obj_res = df["objects"][i].split(',')
            for obj in obj_res:
                if obj.split(':')[0] not in detected_object and obj.split(':')[0] in te.columns_:
                    detected_object.append(obj.split(':')[0])
    val_dataset.append(detected_object)

custom_dataset_train = Dataset.from_dict(CustomDataset(dataset=train_dataset, te=te, target_key=-1))

custom_dataset_val = {"text":[], "label":[]}
with open("../../result/Youtube-8M_seg/selected_keys_10_unvague_Youtube.txt", 'rb') as f:
    key_candidates = pickle.load(f)
for target_key in key_candidates:
    val_group = CustomDataset(dataset=val_dataset, te=te, target_key=target_key)
    custom_dataset_val["text"] += val_group["text"]
    custom_dataset_val["label"] += val_group["label"]
custom_dataset_val = Dataset.from_dict(custom_dataset_val)

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

def tokenize_function(examples):
    return tokenizer(examples["text"], truncation=True, max_length=512)

training_data = custom_dataset_train.map(tokenize_function, batched=True, load_from_cache_file=False)
val_data = custom_dataset_val.map(tokenize_function, batched=True, load_from_cache_file=False)

model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=1)
training_args = TrainingArguments(output_dir="../../model/train_youtube", evaluation_strategy="steps", warmup_ratio=0.1, logging_steps=500, per_device_train_batch_size=128, per_device_eval_batch_size=512, num_train_epochs=4, learning_rate=2e-05, load_best_model_at_end = True, metric_for_best_model = 'avg processed videos', greater_is_better = False, save_total_limit=1)

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
print("training time is ", time.time()-start)
