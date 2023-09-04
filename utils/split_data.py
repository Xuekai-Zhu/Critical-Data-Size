import pandas as pd
import json
import random


def load_json_file(in_file, return_dict=False):
    with open(in_file, "r") as f:
        data = f.readlines()
    if return_dict:
        all_data = [json.loads(i) for i in data]
        return all_data
    else:
        return data


def save_to_json(ratio, original_file, save_dir):
    original_data = load_json_file(original_file)
    # shuffle
    random.shuffle(original_data)
    # extract
    n = int(ratio * len(original_data))
    extract_data = original_data[:n]  # slice the list to get the first n elements
    # save
    save_file = save_dir + "/{}_train.json".format(ratio)
    with open(save_file, "w") as f:
        for i in extract_data:
            f.write(i)

save_dir = "../datasets/yesno_task/random_pruning"
original_file = "../datasets/yesno_task/datatsets/train.json"

for i in range(10):
    ratio = 0.1*(i+1)
    save_to_json(ratio, original_file, save_dir)

