import time
import pandas as pd
import os
import numpy as np
import pickle
import itertools
import gensim
from mlxtend.preprocessing import TransactionEncoder
from nltk.corpus import wordnet

numberbatch_model = gensim.models.KeyedVectors.load_word2vec_format('../data/ConceptNet/numberbatch-en-19.08.txt', binary=False)

# dataset_name = "Youtube-8M_seg"
dataset_name = "HowTo100M"
train_dir = "../data/"+dataset_name+"/video_label/yolo-9000-clip-select-split/train/"
train_dataset = []   # video-level objects
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

test_true_dir = "../data/"+dataset_name+"/video_label/yolo-9000-clip-select-split/test_true/"
test_true_dataset = []
for test_true_label in os.listdir(test_true_dir):
    label_path = test_true_dir + test_true_label
    df = pd.read_csv(label_path)

    true_object = []
    for i in range(len(df["objects"])):
        if type(df["objects"][i]) != np.float64 and type(df["objects"][i]) != float:
            obj_res = df["objects"][i].split(',')
            for obj in obj_res:
                if obj.split(':')[0] not in true_object and obj.split(':')[0] in te.columns_:
                    true_object.append(obj.split(':')[0])
    test_true_dataset.append(true_object)

obj_freq = {}
for true_object in test_true_dataset:
    for comb in itertools.combinations(true_object, 2):
        if comb in obj_freq:
            obj_freq[comb] += 1
        elif (comb[1], comb[0]) in obj_freq:
            obj_freq[(comb[1], comb[0])] += 1
        else:
            obj_freq[comb] = 1

# with open("../model/frequency_2obj_Youtube.txt", 'wb') as f:
with open("../model/frequency_2obj_How.txt", 'wb') as f:
    pickle.dump(obj_freq, f)

# with open("../model/frequency_2obj_Youtube.txt", 'rb') as f:
with open("../model/frequency_2obj_How.txt", 'rb') as f:
    obj_freq = pickle.load(f)

selected_keys = []
for comb in obj_freq:
    if obj_freq[comb] >= 150 and comb[0].replace(" ", "_") in numberbatch_model and comb[1].replace(" ", "_") in numberbatch_model:
        selected_keys.append((te.columns_.index(comb[0]), te.columns_.index(comb[1])))


entity = wordnet.synsets("entity")[0]
physical_entity = entity.hyponyms()[1]
object01 = physical_entity.hyponyms()[2] # 'object.n.01'

vague_objects = set()
max_len = 3
synset_queue = [(object01,0)]

while len(synset_queue) > 0:
    cur_synset, cur_len = synset_queue.pop(0)
    if cur_len+1 <= max_len:
        hyponyms = cur_synset.hyponyms()
        for hyponym in hyponyms:
            if hyponym.name().split('.')[1] == 'n':
                synset_queue.append((hyponym, cur_len+1))
    lemmas = cur_synset.lemmas()
    for lemma in lemmas:
        vague_objects.add(lemma.name())

for selected_key in selected_keys:
    if te.columns_[selected_key[0]].replace(" ", "_") in vague_objects or te.columns_[selected_key[1]].replace(" ", "_") in vague_objects:
        selected_keys.remove(selected_key)

# with open("../result/Youtube-8M_seg/selected_2keys_high_unvague_Youtube.txt", 'wb') as f:
with open("../result/HowTo100M/selected_2keys_high_unvague_How.txt", 'wb') as f:
    pickle.dump(selected_keys, f)