import pandas as pd
import os
import numpy as np
from random import sample
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence

import sys
sys.path.append('../')

# from conditional_prob.train_prob import train_prob

class CustomDatasetForCombine(Dataset):
    def __init__(self, train_dir, dataset, te, target_key, numberbatch_model, train_cond_prob_dic=None):
        self.feature = []
        self.label = []
        if target_key == -1:
            for cur_tuple in dataset:
                target_candidates = cur_tuple + sample(te.columns_, len(cur_tuple))
                for target_obj in target_candidates:
                    cur_feature = []
                    for obj in cur_tuple:
                        if obj != target_obj:
                            if target_obj.replace(" ", "_") in numberbatch_model and obj.replace(" ", "_") in numberbatch_model:
                                cur_feature.append(np.concatenate((np.array([numberbatch_model.similarity(target_obj.replace(" ", "_"), obj.replace(" ", "_"))]), np.array([train_cond_prob_dic[target_obj][obj]])))) # len=2
                            else:
                                cur_feature.append(np.concatenate((np.array([0]), np.array([train_cond_prob_dic[target_obj][obj]])))) # len=2
                    self.feature.append(torch.Tensor(cur_feature))
                    if target_obj in cur_tuple:
                        self.label.append(1)
                    else:
                        self.label.append(0)               
                    
        else:
            # train_cond_prob = train_prob(train_dir=train_dir, frame_interval=1, video_interval=1, te=te, target_key=target_key)

            for cur_tuple in dataset:
                cur_feature = []
                for obj in cur_tuple:
                    if obj != te.columns_[target_key]:
                        # cur_feature.append(np.concatenate((target_embedding, obj_embedding, np.array([train_cond_prob[obj]]))))
                        if te.columns_[target_key].replace(" ", "_") in numberbatch_model and obj.replace(" ", "_") in numberbatch_model:
                            cur_feature.append(np.concatenate((np.array([numberbatch_model.similarity(te.columns_[target_key].replace(" ", "_"), obj.replace(" ", "_"))]), np.array([train_cond_prob_dic[te.columns_[target_key]][obj]]))))
                        else:
                            cur_feature.append(np.concatenate((np.array([0]), np.array([train_cond_prob_dic[te.columns_[target_key]][obj]]))))
                self.feature.append(torch.Tensor(cur_feature))
                if te.columns_[target_key] in cur_tuple:
                    self.label.append(1)
                else:
                    self.label.append(0)

    def __len__(self):
        return(len(self.label))
    
    def __getitem__(self, idx):
        sample = {"feature": self.feature[idx], "label": self.label[idx]}
        return(sample)

class RNN(nn.Module):
    def __init__(self, input_size, hidden_dim, n_layers):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_dim, num_layers=n_layers, batch_first=True, bidirectional=True)
        self.drop = nn.Dropout(p=0.8)
        self.fc = nn.Linear(2*hidden_dim, 1)

    def forward(self, x, x_len):
        packed_input = pack_padded_sequence(x, x_len, batch_first=True, enforce_sorted=False)
        packed_output, _ = self.lstm(packed_input)
        output, _ = pad_packed_sequence(packed_output, batch_first=True)

        out_forward = output[range(len(output)), x_len-1, :self.hidden_dim]
        out_reverse = output[:, 0, self.hidden_dim:]
        out_reduced = torch.cat((out_forward, out_reverse),1)
        out_drop = self.drop(out_reduced)

        out_fc = self.fc(out_drop)
        out_fc = torch.squeeze(out_fc, 1)
        out = torch.sigmoid(out_fc)

        return out

def train(model, iterator, optimizer, criterion, device):
    epoch_loss = 0

    model.train()

    for idx, (sorted_x_pad, sorted_y, sorted_x_lens, sorted_idx) in enumerate(iterator):
        optimizer.zero_grad()
        sorted_x_pad = sorted_x_pad.to(device)
        sorted_y = sorted_y.to(device)
        predictions = model(sorted_x_pad, sorted_x_lens)
        loss = criterion(predictions,sorted_y)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    
    return(epoch_loss/len(iterator))

def evaluate(model, iterator, criterion, device):
    epoch_loss = 0
    
    model.eval()

    with torch.no_grad():
        for idx, (sorted_x_pad, sorted_y, sorted_x_lens, sorted_idx) in enumerate(iterator):
            sorted_x_pad = sorted_x_pad.to(device)
            sorted_y = sorted_y.to(device)
            predictions = model(sorted_x_pad, sorted_x_lens)
            loss = criterion(predictions, sorted_y)
            epoch_loss += loss.item()
        
    return(epoch_loss/len(iterator))                         

def my_collate(batch):
    xx = []
    yy = []
    for i in range(len(batch)):
        xx.append(batch[i]['feature'])
        yy.append(batch[i]['label'])
    x_lens = [len(x) for x in xx]
    x_pad = pad_sequence(xx, batch_first = True, padding_value = 0)

    idx = sorted(range(len(x_lens)), key=lambda k: x_lens[k], reverse=True)
    sorted_x_lens = torch.zeros(len(x_lens))
    sorted_x_pad = torch.zeros(len(x_pad), len(x_pad[0]), len(x_pad[0][0]))
    sorted_y = torch.zeros(len(yy))
    for i in range(len(idx)):
        sorted_x_lens[i] = x_lens[idx[i]]
        sorted_x_pad[i] = x_pad[idx[i]]
        sorted_y[i] = yy[idx[i]]
    sorted_x_lens = sorted_x_lens.long()

    return sorted_x_pad, sorted_y, sorted_x_lens, idx