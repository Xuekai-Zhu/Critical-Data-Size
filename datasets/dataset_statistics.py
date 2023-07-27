import os
import json 
import csv

folder_path = "./natural-instructions-2.8/yesno_task/datatsets/"
info_list= []
# 逐一检查文件夹中的文件
for filename in os.listdir(folder_path):
    file_path = os.path.join(folder_path, filename)

    # 计算文件中的行数
    with open(file_path, 'r') as f:
        data = f.readlines()
        lines = len(data)
        output = json.loads(data[0])["output"]

    # 将文件名和行数加入到列表中
    info_list.append([filename, lines])

output_csv = "./natural-instructions-2.8/yesno_task/datatsets/statistic.csv"
# 将结果写入CSV文件
with open(output_csv, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['filename', "size"])  # 写入标题
    writer.writerows(info_list)  # 写入数据