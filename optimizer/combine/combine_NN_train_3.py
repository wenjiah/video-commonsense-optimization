import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
import os
import numpy as np
import random
import pickle
import time
import gensim
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from combine_NN_3 import *           

numberbatch_model = gensim.models.KeyedVectors.load_word2vec_format('/z/wenjiah/query_commonsense/data/ConceptNet/numberbatch-en-19.08.txt', binary=False)
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

val_dataset = []
val_dir = "/z/wenjiah/query_commonsense/data/Youtube-8M_seg/video_label/yolo-9000-clip-select-split/val/"
frame_interval = 1
video_interval = 1
label_count = 0
for val_label in os.listdir(val_dir):
    label_count += 1
    if label_count%video_interval == 0:
        label_path = val_dir + val_label
        df = pd.read_csv(label_path)

        detected_object = []
        for i in range(len(df["objects"])):
            if i%frame_interval == 0 and type(df["objects"][i]) != np.float64 and type(df["objects"][i]) != float:
                obj_res = df["objects"][i].split(',')
                for obj in obj_res:
                    if obj.split(':')[0] not in detected_object and obj.split(':')[0] in te.columns_:
                        detected_object.append(obj.split(':')[0])
        val_dataset.append(detected_object)
random.shuffle(val_dataset)

training_data = CustomDatasetForCombine(train_dir=train_dir, dataset=val_dataset[:int(0.8*len(val_dataset))], te=te, target_key=-1, numberbatch_model=numberbatch_model, train_cond_prob_dic=train_cond_prob_dic)
val_data = CustomDatasetForCombine(train_dir=train_dir, dataset=val_dataset[int(0.8*len(val_dataset)):], te=te, target_key=-1, numberbatch_model=numberbatch_model, train_cond_prob_dic=train_cond_prob_dic)

train_dataloader = DataLoader(training_data, batch_size=64, shuffle=True, collate_fn=my_collate)
val_dataloader = DataLoader(val_data, batch_size=64, shuffle=True, collate_fn=my_collate)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model = RNN(input_size=2, hidden_dim=256, n_layers=3)
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.BCELoss()

model = model.to(device)
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
            torch.save(model.state_dict(), '../../temp_result/combine_model_3.pt')
        
        print("Train loss:", train_loss)
        print("Validation loss:", val_loss)

print("Total time: ", time.time()-start)
print("********************")