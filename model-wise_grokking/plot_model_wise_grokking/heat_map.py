import matplotlib.pyplot as plt
import seaborn as sns
import os, re
import json
import numpy as np
from tqdm import tqdm
from scipy.ndimage import gaussian_filter
import torch

def find_train_logs(base_dir, file_name):
    sub_dir_1 = [f"IMDB_{i:.1f}" for i in np.arange(0.1, 1, 0.1)]
    sub_sub_dir_2 = [f"hidden_size_{i}" for i in np.arange(16, 272, 16)]
    all_files = []

    for sub in sub_dir_1:
        for sub_sub in sub_sub_dir_2:
            file_path = "/".join([base_dir, sub, sub_sub, file_name])
            # if "1.0" not in file_path:
            all_files.append(file_path)
    return all_files

def extract_final_acc(train_log, key=None, step=None):
    log_ist = []
    with open(train_log, "r") as f:
        data = f.readlines()
        for i in data:
            if key in i:
                json_str = i.split("INFO:root:")[-1]
                log_ist.append(json.loads(json_str))
    
    if step is not None:
        for i in log_ist:
            if i["Step"] == step:
                return i
    else:
        return log_ist

def plot_model_wise_grokking(in_dir, data_size=None, hidden_size=None):
    all_train_logs = find_train_logs(in_dir, "train_log.txt")

    all_final_acc = []
    for i in tqdm(all_train_logs):
        acc = extract_final_acc( i, key="Test Acc", step=400000)
        all_final_acc.append(acc["Test Acc"])
    
    all_final_acc = np.array(all_final_acc).reshape(len(data_size), -1)
    
    # heatmap(all_final_acc, save_file="model-wise_grokking/plot_model_wise_grokking/figures/test_acc.png",
    #         data_size=data_size,
    #         hidden_size=hidden_size)
    
def heatmap_diferent_hidden_size(value, save_file=None, data_size=None, y_lable=None):
    # Set seaborn style
    sns.set_theme(style="white")
    
    # Initialize the figure
    fig, ax = plt.subplots(figsize=(12, 6))
    custom_cmap = sns.color_palette("rocket", as_cmap=True)
    # Create heatmap
    sns.heatmap(
        value, 
        cmap=custom_cmap, 
        linewidths=0.1,
        yticklabels=y_lable if y_lable is not None else "auto",
        cbar_kws={'label': 'Acc'},
        ax=ax
    )
    
    # Set x and y labels
    ax.set_xlabel("Steps (1e3)", fontsize=15)
    ax.set_ylabel("Hidden Size", fontsize=15)

    # Save the figure
    plt.savefig(save_file, dpi=500, bbox_inches="tight")
    plt.close(fig)
    
def heatmap_diferent_data_frac(value, save_file=None, data_size=None, y_lable=None):
    # Set seaborn style
    sns.set_theme(style="white")
    
    # Initialize the figure
    fig, ax = plt.subplots(figsize=(12, 6))
    custom_cmap = sns.color_palette(
                                    # "rocket",
                                    "magma_r",
                                    as_cmap=True)
    # Create heatmap
    sns.heatmap(
        value, 
        cmap=custom_cmap, 
        linewidths=0.1,
        yticklabels=y_lable if y_lable is not None else "auto",
        cbar_kws={'label': 'Acc'},
        ax=ax
    )
    # Add contour lines
    # plt.contour(value, levels=[0.825], colors='white', linestyles='dashed', linewidths=0.5)


    # Set x and y labels
    ax.set_xlabel("Steps (1e3)", fontsize=15)
    ax.set_ylabel("Data Frac.", fontsize=15)

    # Save the figure
    plt.savefig(save_file, dpi=500, bbox_inches="tight")
    plt.close(fig)

def heatmap_hidden_size(in_dir):
    data_size = [i for i in np.arange(0.1, 1.1, 0.1)]
    all_data_frac = [f"IMDB_{i:.1f}" for i in np.arange(0.1, 1.1, 0.1)]
    all_hidden_size = [f"hidden_size_{i}" for i in np.arange(16, 272, 16)]
    
    # all_train_logs = find_train_logs(in_dir, "train_log.txt")
    for data_frac in all_data_frac:
        all_subdata_acc = []
        for hidden_size_i in tqdm(all_hidden_size):
            file_path = "/".join([in_dir, data_frac, hidden_size_i, "train_log.txt"])
            single_frac_acc = extract_final_acc(file_path, key="Test Acc", 
                                                # step=400000
                                                )
            test_acc = [i["Test Acc"] for i in single_frac_acc]
            line = gaussian_filter(test_acc, sigma=10)
            all_subdata_acc.append(line)
            
        # x, y, highest_x, highest_y, num_colors = obtain_data(all_subdata_acc, data_size=data_size)
        
        save_dir = os.path.join("model-wise_grokking/plot_model_wise_grokking/figures/heat_map", f"{data_frac}")
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
            
        save_png = os.path.join(save_dir, f"{hidden_size_i}.png")
        all_final_acc = np.array(all_subdata_acc)
        # flipped_matrix = np.flipud(all_final_acc.T)
        heatmap_diferent_hidden_size(all_final_acc, save_file=save_png, 
                y_lable=[i for i in np.arange(16, 272, 16)],
                # hidden_size=
                )
        
def heatmap_data_frac(in_dir):
    data_size = [i for i in np.arange(0.1, 1.1, 0.1)]
    all_data_frac = [f"IMDB_{i:.1f}" for i in np.arange(0.1, 1.1, 0.1)]
    all_hidden_size = [f"hidden_size_{i}" for i in np.arange(16, 272, 16)]
    
    # all_train_logs = find_train_logs(in_dir, "train_log.txt")
    for hidden_size_i in tqdm(all_hidden_size):
        all_subdata_acc = []
        for data_frac in all_data_frac:
            file_path = "/".join([in_dir, data_frac, hidden_size_i, "train_log.txt"])
            single_frac_acc = extract_final_acc(file_path, key="Test Acc", 
                                                # step=400000
                                                )
            test_acc = [i["Test Acc"] for i in single_frac_acc]
            line = gaussian_filter(test_acc, sigma=50)
            all_subdata_acc.append(line)
            
        # x, y, highest_x, highest_y, num_colors = obtain_data(all_subdata_acc, data_size=data_size)
        
        save_dir = os.path.join("model-wise_grokking/plot_model_wise_grokking/figures/heat_map/data_frac", f"{hidden_size_i}")
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
            
        save_png = os.path.join(save_dir, f"{data_frac}.png")
        all_final_acc = np.array(all_subdata_acc)
        # flipped_matrix = np.flipud(all_final_acc.T)
        heatmap_diferent_data_frac(all_final_acc, save_file=save_png, 
                y_lable=[round(i, 1) for i in np.arange(0.1, 1.1, 0.1)],
                )


def heat_map_for_grokking(in_dir, key, save_file=None):
    data_size = [i for i in np.arange(0.1, 1.1, 0.1)]
    data_frac = [f"IMDB_{i:.1f}" for i in np.arange(0.1, 1.1, 0.1)]
    hidden_size = [f"hidden_size_{i}" for i in np.arange(16, 272, 16)]
    
    # all_train_logs = find_train_logs(in_dir, "train_log.txt")
    all_acc = []
    for i_hidden_size in tqdm(hidden_size):
        all_subdata_acc = []
        for i_data_frac in data_frac:
            file_path = "/".join([in_dir, i_data_frac, i_hidden_size, "train_log.txt"])
            single_frac_acc = extract_final_acc(file_path, key="Test Acc", 
                                                # step=400000
                                                )
            if key == "Test Acc":
            
                test_acc = [i["Test Acc"] for i in single_frac_acc]
            else:
                test_acc = [i["Train Acc"] for i in single_frac_acc]
            line = gaussian_filter(test_acc, sigma=50)
            all_subdata_acc.append(max(line))
        all_acc.append(all_subdata_acc)
    
    acc = np.array(all_acc) # 16(16 ~ 256 hidden size) x 10 (0.1 ~ 1.0 data frac)
    plot_scale = 10
    train_large = torch.nn.functional.interpolate(torch.tensor(acc).unsqueeze(dim=0).unsqueeze(dim=0), 
                                                  scale_factor=(plot_scale,plot_scale), mode='bilinear')[0,0].detach().numpy()
    # Set seaborn style
    sns.set_theme(style="white")
    num_rows, num_cols = train_large.shape
    
    # Initialize the figure
    fig, ax = plt.subplots(figsize=(12, 6))
    custom_cmap = sns.color_palette(
                                    # "rocket",
                                    "magma_r",
                                    as_cmap=True)
    # Create heatmap
    sns_heatmap = sns.heatmap(
        train_large, 
        # cmap=custom_cmap, 
        # linewidths=0.1,
        # yticklabels=[i for i in np.arange(16, 272, 16)],
        # xticklabels=[i for i in np.arange(0.1, 1.1, 0.1)],
        cbar_kws={'label': 'Acc'},
        ax=ax
    )
    # 设置颜色条标签的字体大小
    cbar = sns_heatmap.collections[0].colorbar
    cbar.ax.set_ylabel('Acc', fontsize=20)

    # 设置颜色条刻度标签的字体大小
    cbar.ax.tick_params(labelsize=16)
    # plt.imshow(train_large, cmap='Reds', aspect=0.7)
    # plt.colorbar()  # 如果需要颜色条
    # plt.savefig('/path/to/your/image.png')  # 保存图像
    
    # Add contour lines
    # plt.contour(train_large, levels=[0.8], colors='white', linestyles='dashed', linewidths=0.5)
    # CS = plt.gca().contour(X, Y, train_large, [0.8], colors=["white"], linestyles=["dashed"])
    # plt.gca().clabel(CS, inline=True, fontsize=10)
    
    # 生成等高线的坐标网格
    x = np.linspace(0, num_cols - 1, num_cols)
    y = np.linspace(0, num_rows - 1, num_rows)
    X, Y = np.meshgrid(x, y)

    # 绘制等高线
    CS = plt.gca().contour(X, Y, train_large, [0.75, 0.775, 0.8], colors=["white", "white", "white"], linestyles=["dashed", "dashed", "dashed"])
    plt.gca().clabel(CS, inline=True, fontsize=15)
    

    # 假设的刻度位置和标签
    x_ticks = [0, num_cols // 2, num_cols - 1]
    y_ticks = [0, num_rows // 2, num_rows - 1]

    # 对应的刻度标签
    y_labels = [16, 128, 256]
    x_labels = [0.1, 0.5, 1]
    
    # 设置 x 轴和 y 轴的刻度及其标签
    ax.set_xticks(x_ticks)
    ax.set_xticklabels(x_labels, fontsize=16)  # 设置字体大小
    ax.set_yticks(y_ticks)
    ax.set_yticklabels(y_labels, fontsize=16)

    # Set x and y labels
    ax.set_xlabel("Data Frac.", fontsize=20)
    ax.set_ylabel("Hidden Size", fontsize=20)

    # Save the figure
    plt.savefig(save_file, dpi=500, bbox_inches="tight")
    plt.close(fig)
            
            
            
if __name__ == '__main__':
    # v1
    # base_dir = "model-wise_grokking/model/encoder"
    # # subdatat_size = [9, 16]
    # sub_data_size = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    # hidden_size = [i for i in np.arange(16, 272, 16)]
    # plot_model_wise_grokking(base_dir, data_size=sub_data_size, hidden_size=hidden_size)
    
    # v2
    base_dir = "model-wise_grokking/model/encoder"
    # heatmap_hidden_size(base_dir)
    
    # heatmap_data_frac(base_dir)
    
    heat_map_for_grokking(base_dir,"Test Acc", "model-wise_grokking/plot_model_wise_grokking/figures/heat_map/test_acc_heatmap_data_frac_vs_hiddensize.png")
    
    