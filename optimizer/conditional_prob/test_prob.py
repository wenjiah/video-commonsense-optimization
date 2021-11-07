import numpy as np

def test_prob(test_dataset, test_true_dataset, te, target_key):
    test_cond_prob = {}
    for obj in te.columns_:
        count_obj = 0
        count_tar = 0
        for k in range(len(test_dataset)):
            test_object = test_dataset[k]
            if obj in test_object:
                count_obj += 1
                if te.columns_[target_key] in test_true_dataset[k]:
                    count_tar += 1
        if count_obj > 0:
            test_cond_prob[obj] = count_tar/count_obj
    return test_cond_prob
