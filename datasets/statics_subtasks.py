import os
import csv
import json

# 定义文件夹路径
folder_path = "./natural-instructions-2.8/sub_classification_tasks/" 

# 定义输出的CSV文件
output_csv = './natural-instructions-2.8/output.csv' 

# 初始化一个空列表来保存数据
info_list = []

# 逐一检查文件夹中的文件
for filename in os.listdir(folder_path):
    file_path = os.path.join(folder_path, filename)

    # 计算文件中的行数
    with open(file_path, 'r') as f:
        data = f.readlines()
        lines = len(data)
        output = json.loads(data[0])["output"]

    # 将文件名和行数加入到列表中
    info_list.append([filename, lines, output])

# 将结果写入CSV文件
with open(output_csv, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['filename', 'num_lines', "answer"])  # 写入标题
    writer.writerows(info_list)  # 写入数据
