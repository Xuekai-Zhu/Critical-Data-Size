import json
import numpy as np
from tqdm import tqdm
import random

import numpy as np
import json
import random
import os

def create_modular_dataset(P=113):
    a_values = np.arange(0, P)
    b_values = np.arange(0, P)

    # Construct dataset
    data = []
    for a in a_values:
        for b in b_values:
            result = (a + b) % P
            data.append((a, b, result))

    # Convert to numpy array for convenience
    data_array = np.array(data)

    # Convert numpy int64 to native Python int
    data_json_list = [json.dumps({"text": f"{row[0]} + {row[1]} % {P} = ", "target": f"{int(row[2])}"}) for row in data_array]
    return data_json_list

def split_and_save_dataset(data_json_list, train_filename, test_filename, ratio=0.75):
    # Split dataset
    clip_index = int(len(data_json_list) * ratio)
    random.shuffle(data_json_list)
    train_set = data_json_list[:clip_index]
    test_set = data_json_list[clip_index:]
    
    # Save train set
    with open(train_filename, 'w') as f:
        for item in train_set:
            f.write(item + "\n")
            
    # Save test set
    with open(test_filename, 'w') as f:
        for item in test_set:
            f.write(item + "\n")

def create_vocab_file(P=113, filename="grokking/modular/vocab.txt"):
    tokens = [str(i) for i in range(P)] + ["+", "=", "%"]
    with open(filename, 'w') as f:
        for token in tokens:
            f.write(token + "\n")
    # return filename

if __name__ == '__main__':
    
    # P = 113
    # Example usage
    # data = create_modular_dataset()
    # with open("grokking/modular/modular_data.json", "w") as f:
    #     for token in data:
    #             f.write(token + "\n")


    # train_filename = "grokking/modular/train_modular_data.json"
    # test_filename = "grokking/modular/test_modular_data.json"
    # split_and_save_dataset(data, train_filename, test_filename)

    # create_vocab_file()

    # P = 97
    data = create_modular_dataset(P=97)
    with open("grokking/modular/P=97_datasets/modular_data.json", "w") as f:
        for token in data:
                f.write(token + "\n")
                
    train_filename = "grokking/modular/P=97_datasets/train_modular_data.json"
    test_filename = "grokking/modular/P=97_datasets/test_modular_data.json"
    split_and_save_dataset(data, train_filename, test_filename, ratio=0.5)
    
    vacab_file = "grokking/modular/P=97_datasets/vocab.txt"
    vocab_file = create_vocab_file(P=97, filename=vacab_file)



