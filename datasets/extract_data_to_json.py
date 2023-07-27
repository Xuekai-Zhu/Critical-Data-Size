from datasets import load_dataset
import os
import glob
import json
from tqdm import tqdm

# search json file 
def find_json_files(directory):
    return glob.glob(os.path.join(directory, '*.json'))

directory = './natural-instructions-2.8/tasks/'  # 更改为你需要搜索的文件夹路径
json_files = find_json_files(directory)
classification_task = [i for i in json_files if "class" in i]

# with open("super_ni_classification_task.txt", "w") as f:
    # for i in classification_task:
        # f.write(i + "\n")
        
# new_format = {
#     "instruction": ,
#     "input": , 
#     "output":,}

data_list = []
for i in tqdm(classification_task):
    with open(i, "r") as f:
        data = json.load(f)
        instruction = data["Definition"]
        instances = data["Instances"]
        new_save_path = i.replace("tasks", "sub_classification_tasks")
        with open(new_save_path, "w") as f_out:
            for j in instances:
                j["instruction"] = instruction
                f_out.write(json.dumps(j) + "\n")

