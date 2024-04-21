import json
import random
from tqdm import tqdm
import os

def load_json_file(in_file, return_dict=False):
    with open(in_file, "r") as f:
        data = f.readlines()
    if return_dict:
        all_data = [json.loads(i) for i in data]
        return all_data
    else:
        return data
    
def split_data(input_data, save_file, sample_nums=None, ratio=None):
    data = load_json_file(input_data)
    if ratio is not None and ratio < 1:
        sample_num = int(len(data) * ratio)
    elif sample_nums is not None and sample_nums > 1:
        sample_num = sample_nums
    
    fileter_sample = random.sample(data, sample_num)
    with open(save_file, "w") as f:
        for i in tqdm(fileter_sample):
            f.write(i)
    
if __name__ == '__main__':
    
    # p = 113
    orignal_data = "grokking/modular/p=113_datasets/train_modular_data.json"
    save_dir = "grokking/modular/p=113_datasets/subsets_v2"
    for num in range(1000, 10000, 500):
        save_file = os.path.join(save_dir, f"{num}_train.json")
        split_data(orignal_data,save_file, sample_nums=num)
        
    # # p = 97
    # orignal_data = "grokking/modular/P=97_datasets/train_modular_data.json"
    # save_dir = "grokking/modular/P=97_datasets/subsets"
    # # for num in range(1000, 9000, 1000):
    # num = 0.5
    # save_file = os.path.join(save_dir, f"{num}_train.json")
    # split_data(orignal_data,save_file, ratio=num)