import numpy as np
from random import sample

def CustomDataset(dataset, te, target_key):
    custom_dataset = {}
    text_list = []
    label_list = []

    if target_key == -1:
        for video_objects in dataset:
            target_candidates = video_objects + sample(te.columns_, len(video_objects))
            for target_obj in target_candidates:
                text = ""
                for obj in video_objects:
                    if obj != target_obj:
                        text += obj + " "
                text_list.append([text, target_obj])

                if target_obj in video_objects:
                    label_list.append(float(1))
                else:
                    label_list.append(float(0))

    else:
        for video_objects in dataset:
            text = ""
            for obj in video_objects:
                if obj != te.columns_[target_key]:
                    text += obj + " "
            text_list.append([text, te.columns_[target_key]])

            if te.columns_[target_key] in video_objects:
                label_list.append(float(1))
            else:
                label_list.append(float(0))

    custom_dataset["text"] = text_list
    custom_dataset["label"] = label_list
    return custom_dataset

def ComputeMetric(eval_pred):
    logits, labels = eval_pred

    limit = 0.2
    target_count = 445
    logits = logits.reshape(target_count, -1)
    labels = labels.reshape(target_count, -1)

    process_count = 0
    perfect_count = 0
    for group_idx in range(target_count):
        logits_group = logits[group_idx]
        labels_group = labels[group_idx]
        labels_group = labels_group[np.argsort(-logits_group)]

        satisfy_num = labels_group.sum()
        perfect_count += satisfy_num*limit
        labels_group_cumsum = np.cumsum(labels_group)
        for idx in range(len(labels_group_cumsum)):
            if labels_group_cumsum[idx] >= satisfy_num*limit:
                process_count += idx+1
                break

    return {"avg processed videos": process_count/target_count}

