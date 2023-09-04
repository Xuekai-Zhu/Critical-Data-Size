import random
import os
import json
from tqdm import tqdm

data_dir = "./yesno_task/orignal_task/"
all_files = os.listdir(data_dir)

all_train = []
all_vaild = []
all_test = []
for i in tqdm(all_files):
    file_path = os.path.join(data_dir, i)
    with open(file_path, "r") as f:
        now_data = f.readlines()
        lower_answer_data = []
        for j in now_data:
            item = json.loads(j)
            try:
                assert len(item["output"]) == 1
                answer = item["output"][0].lower()
                assert answer in ["yes", "no"]
            except:
                print(file_path)
                print(item["output"])
                continue
            item["output"] = item["output"][0].lower()
            new_ietm = json.dumps(item) + "\n"
            lower_answer_data.append(new_ietm)
            
    # split 
    random.shuffle(lower_answer_data)
    # 计算划分的索引
    total_len = len(lower_answer_data)
    train_index = int(0.9 * total_len)
    val_index = train_index + int(0.05 * total_len)

    # 划分数据
    train_data = lower_answer_data[:train_index]
    val_data = lower_answer_data[train_index:val_index]
    test_data = lower_answer_data[val_index:]
    
    # 添加到全体数据集中
    all_train.extend(train_data)
    all_vaild.extend(val_data)
    all_test.extend(test_data)

print("------")

def save_data(input_data, save_path):
    with open(save_path, "w") as f:
        for i in input_data:
            f.write(i)

save_data(all_train, "./natural-instructions-2.8/yesno_task/datatsets/train.json")
# save_data(all_vaild, "./natural-instructions-2.8/yesno_task/datatsets/valid.json")
# save_data(all_test, "./natural-instructions-2.8/yesno_task/datatsets/test.json")