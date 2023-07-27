import pandas as pd
import shutil
import os

data_path_prefix = "./natural-instructions-2.8/sub_classification_tasks/"

# 定义你的CSV文件路径
csv_file_path = './natural-instructions-2.8/sub-tasks-lengths.csv' 

# 读取CSV文件
df = pd.read_csv(csv_file_path)

# 过滤出符合条件的行：num_lines大于6000并且answer为'yes'或'no'

# filtered_df = df[ (df['num_lines'] > 6000)   & (df['answer'].isin(['yes', 'no']))]

# 创建一个空的DataFrame来保存满足条件的行
filtered_df = []

# 逐行检查DataFrame
for index, row in df.iterrows():
    if row['num_lines'] > 6000 and ("yes" in row['answer'].lower() or "no" in row['answer'].lower()):
        # 如果满足条件，将这一行添加到新的DataFrame中
        filtered_df.append(row["filename"])

final_file = [os.path.join(data_path_prefix, i) for i in filtered_df]


# 目标文件夹路径
destination_folder = "./natural-instructions-2.8/yesno_task/"

for filename in final_file:
    # 如果文件存在
    if os.path.isfile(filename):
        # 移动文件
        shutil.move(filename, destination_folder)
    else:
        print(f"The file {filename} does not exist.")
