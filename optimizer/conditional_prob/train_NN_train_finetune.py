import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
import os
import numpy as np
import random
import pickle
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from train_NN import *           

with open("../../temp_result/train_cond_prob_dic_Youtube.txt", 'rb') as f:
    train_cond_prob_dic = pickle.load(f)

train_dataset = [] 
train_dir = "/z/wenjiah/query_commonsense/data/Youtube-8M_seg/video_label/yolo-9000-clip-select-split/train/"
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

with open("../../temp_result/finetune_videos.txt", 'rb') as f:
    finetune_videos = pickle.load(f)
test_true_dataset = []
test_true_dir = "/z/wenjiah/query_commonsense/data/Youtube-8M_seg/video_label/yolo-9000-clip-select-split/test_true/"
for test_true_label in finetune_videos:
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
random.shuffle(test_true_dataset)

training_data = CustomDatasetForTrain(train_dir=train_dir, dataset=test_true_dataset[:int(0.8*len(test_true_dataset))], te=te, target_key=-1, train_cond_prob_dic=train_cond_prob_dic)
val_data = CustomDatasetForTrain(train_dir=train_dir, dataset=test_true_dataset[int(0.8*len(test_true_dataset)):], te=te, target_key=-1, train_cond_prob_dic=train_cond_prob_dic)

train_dataloader = DataLoader(training_data, batch_size=64, shuffle=True, collate_fn=my_collate)
val_dataloader = DataLoader(val_data, batch_size=64, shuffle=True, collate_fn=my_collate)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model = RNNForTrain(input_size=1, hidden_dim=256, n_layers=3)
model = model.to(device)
model.load_state_dict(torch.load("../../temp_result/training_model_1.pt"))

for param in model.parameters():
    param.requires_grad = False
for param in model.fc.parameters():
    param.requires_grad = True

optimizer = optim.Adam(model.fc.parameters(), lr=0.001)
criterion = nn.BCELoss()
criterion = criterion.to(device)

n_epochs = 20
eval_every = 1
best_val_loss = float('inf')

start = time.time()
for epoch in range(n_epochs):
    train_loss = train(model, train_dataloader, optimizer, criterion, device)

    if (epoch+1)%eval_every == 0:
        val_loss = evaluate(model, val_dataloader, criterion, device)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), '../../temp_result/training_model_1_finetune.pt')
        
        print("Train loss:", train_loss)
        print("Validation loss:", val_loss)

print("Total time: ", time.time()-start)
print("********************")