import numpy as np
from datasets import Dataset
import os
import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
import pickle
import copy
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer, pipeline, DataCollatorWithPadding  

import sys
sys.path.append("../")

from BERT_YouTube.util_2obj_youtube import CustomDataset             

def predict_prob_2obj_bert(test_dataset, te, target_key, model_dir):
    custom_dataset_test = CustomDataset(dataset=test_dataset, te=te, target_key=target_key)
    custom_dataset_test = Dataset.from_dict(custom_dataset_test)

    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

    def tokenize_function(examples):
        return tokenizer(examples["text"], truncation=True, max_length=512)

    test_data = custom_dataset_test.map(tokenize_function, batched=True, load_from_cache_file=False)
    test_data = test_data.remove_columns(["text"])
    test_data.set_format("torch")

    test_dataloader = DataLoader(test_data, batch_size=512, collate_fn=DataCollatorWithPadding(tokenizer=tokenizer))

    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model.to(device)
    model.eval()

    prediction_prob = []
    for batch in test_dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.no_grad():
            outputs = model(**batch)
        logits = outputs.logits
        prob = logits[:,-1].cpu()
        for idx in range(len(prob)):
            if prob[idx] < 0:
                prob[idx] = 0
            if prob[idx] > 1:
                prob[idx] = 1
        prediction_prob += prob

    return prediction_prob