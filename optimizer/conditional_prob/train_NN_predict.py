import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
from conditional_prob.train_NN import *              

def predict_prob_train(train_dir, test_dataset, te, target_key, model_dir, train_cond_prob_dic):
    test_data = CustomDatasetForTrain(train_dir=train_dir, dataset=test_dataset, te=te, target_key=target_key, train_cond_prob_dic=train_cond_prob_dic)
    test_dataloader = DataLoader(test_data, batch_size=64, shuffle=False, collate_fn=my_collate)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = RNNForTrain(input_size=1, hidden_dim=256, n_layers=3)
    model = model.to(device)
    model.load_state_dict(torch.load(model_dir))
    model.eval()

    prediction_prob = []
    with torch.no_grad():
        for idx, (sorted_x_pad, sorted_y, sorted_x_lens, sorted_idx) in enumerate(test_dataloader):
            sorted_x_pad = sorted_x_pad.to(device)
            predictions = model(sorted_x_pad, sorted_x_lens)
            predictions = predictions.to('cpu').numpy()
            probability = np.zeros(len(sorted_idx))
            for i in range(len(sorted_idx)):
                probability[sorted_idx[i]] = predictions[i]
            if prediction_prob == []:
                prediction_prob = probability
            else:
                prediction_prob = np.concatenate((prediction_prob, probability), axis=0)

    return prediction_prob 