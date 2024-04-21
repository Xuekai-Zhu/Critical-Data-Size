from torch.utils.data import Dataset, DataLoader
import torch 
import json
from transformers import AutoTokenizer
import csv
from tqdm import tqdm

class Datasets(Dataset):
    def __init__(self, file):
        # load data
        with open(file, 'r') as f:
            self.seqs = f.readlines()


    def __len__(self):
        """Denotes the total number of samples"""
        return len(self.seqs)

    def __getitem__(self, index):
        data = json.loads(self.seqs[index])
        input_text = data["input"]
        
        instruction = data["instruction"]
        target = data["output"]
        assert isinstance(instruction, list)
        assert isinstance(target, list)
        
        final_input = instruction[0] + input_text
        
        return {"text":final_input, "target":target[0]}


def statics_code_length(file_in, save_out):
    dataset = Datasets(file_in)
    training_generator = DataLoader(dataset, batch_size=1, shuffle=False)
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased", cache_dir="./pre-trained-model/")
    len_list = []
    for text, target in tqdm(training_generator):
        # print(text)
        text_ids = tokenizer(text)["input_ids"][0]
        len_list.append([len(text_ids)])
    
    with open(save_out, 'w', newline='') as f:
        # 创建csv writer对象
        writer = csv.writer(f)
        writer.writerow(["inputs_length"])
        # 将列表数据写入csv文件
        for row in len_list:
            writer.writerow(row)
    
if __name__ == '__main__':
    # dataset = Datasets("./datasets/natural-instructions-2.8/yesno_task/datatsets/test.json")
    # training_generator = DataLoader(dataset, batch_size=8, shuffle=False)
    # for i, (final_input, target) in enumerate(training_generator) :
    #     # print(final_input)
    #     print(target)
    #     if i > 20:
    #         break
    
    train_path = "./datasets/natural-instructions-2.8/yesno_task/datatsets/train.json"
    vaild_path = "./datasets/natural-instructions-2.8/yesno_task/datatsets/valid.json"
    test_path = "./datasets/natural-instructions-2.8/yesno_task/datatsets/test.json"
    
    statics_code_length(train_path, "./plot/statics_files/train_input_length.csv")
    statics_code_length(vaild_path, "./plot/statics_files/vaild_input_length.csv")
    statics_code_length(test_path, "./plot/statics_files/test_input_length.csv")