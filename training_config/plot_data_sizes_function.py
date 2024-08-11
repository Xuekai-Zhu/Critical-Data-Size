import os
from scipy.ndimage import gaussian_filter
import seaborn as sns
import matplotlib.pyplot as plt
import re
import numpy as np
from scipy.optimize import curve_fit

def read_subdirs(input_dir):
    dirs = os.listdir(input_dir)
    files = [os.path.join(input_dir, d, "train_log.txt") for d in dirs]
    files.sort()
    return files

def read_logs(in_file):
    # Pattern to extract the values from each line
    # pattern = re.compile(r"Step: (\d+), Train Loss: ([\d.]+), Test Loss: ([\d.]+), Train Acc: ([\d.]+), Test Acc: ([\d.]+), L2 Norm: ([\d.]+)")
    def log_to_dict(log):
        # 使用正则表达式提取键值对
        pattern = r"([\w\s]+):\s*([\d.]+)"
        matches = re.findall(pattern, log)
        return {key: float(value) for key, value in matches}
    

    # Initialize a dictionary with empty lists for each key
    merged_data = {
        "Step": [],
        "Train Loss": [],
        "Test Loss": [],
        "Train Acc": [],
        "Test Acc": [],
        "L2 Norm": []
    }
    
    # Read the file content
    with open(in_file, 'r') as file:
        content = file.readlines()

    # Extract data from each line using the regex pattern and populate the lists
    for line in content:
        # match = pattern.search(line)
        dic = log_to_dict(line)
        for key in dic.keys():
            merged_data[key.strip()].append(dic[key])
        
        # if matches:
        #     step, train_loss, test_loss, train_acc, test_acc, l2_norm = match.groups()
        #     merged_data["Step"].append(int(step))
        #     merged_data["Train Loss"].append(float(train_loss))
        #     merged_data["Test Loss"].append(float(test_loss))
        #     merged_data["Train Acc"].append(float(train_acc))
        #     merged_data["Test Acc"].append(float(test_acc))
        #     merged_data["L2 Norm"].append(float(l2_norm))

    return merged_data
        
    
    
def plot_multip_lines(files, save_file=None, key=None, clip=6000, sigma=100):

    vaule_dict = {}
    
    for f in files:
        log = read_logs(f)
        vaules = log[key]
        if clip:
            vaules = vaules[:clip]
        line = gaussian_filter(vaules, sigma=sigma)
        numbers = re.findall(r"(\d+)", f)
        if len(numbers) >= 1:
            dir_name = numbers[-1]
        else:
            dir_name = "all_data"
        
        vaule_dict[dir_name] = line
        
    # Remove specific keys
    for unwanted_key in ["all_data", "9500", "9000", "8500", "8000", "7500"]:
        if unwanted_key in vaule_dict:
            vaule_dict.pop(unwanted_key, None)
    
    dataset_sizes = sorted(list(vaule_dict.keys()))

    # 使用 Seaborn 的默认样式
    sns.set_style("whitegrid")
    
    # 使用 Seaborn 的颜色
    # colors = sns.color_palette("blend:#7AB,#EDA", n_colors=len(dataset_sizes))
    colors = sns.color_palette("flare", n_colors=len(dataset_sizes))

    plt.figure(figsize=(10, 6))

    for idx, size in enumerate(dataset_sizes):
        plt.plot(vaule_dict[size], label=f'{size}', color=colors[idx], linewidth=2)

    
    
    plt.xlabel("Steps", fontsize=15)
    plt.ylabel(f"{key}",  fontsize=15)
    plt.tick_params(axis='y', labelsize=15)
    plt.tick_params(axis='x', labelsize=15)
    plt.legend()
    plt.legend(title="Data Size", fontsize=10)  # Added title to the legend here
    plt.tight_layout()
    plt.savefig(save_file, dpi=500, bbox_inches="tight")

def get_thresholds_points(sequence, thresholds):
    # thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    # Finding the indices of values just after each threshold
    # indices = [np.argmax(sequence >= threshold) for threshold in thresholds]

    indices = [np.argmax(sequence >= threshold) if np.any(sequence >= threshold) else 0 for threshold in thresholds]

    # Extracting the values corresponding to the found indices
    # if sequence == 0:
    #     values = np.zeros((len(indices)))
    # else:
    values = sequence[indices]
    
    return indices, values

def scatter_plot(files, save_file=None, key=None, clip=6000, sigma=100, y_max=None):
    thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

    value_dict = {}
    for f in files:
        log = read_logs(f)
        values = log[key]
        if clip:
            values = values[:clip]
        line = gaussian_filter(values, sigma=sigma)
        numbers = re.findall(r"(\d+)", f)
        if len(numbers) >= 1:
            dir_name = numbers[-1]
        else:
            dir_name = "all_data"
        value_dict[dir_name] = line

    # Remove specific keys
    for unwanted_key in ["all_data", "9500", "9000", "8500", "8000", "7500"]:
        if unwanted_key in value_dict:
            value_dict.pop(unwanted_key, None)

    x, y, num_colors = [], [], []
    highest_x, highest_y = [], []
    for item, line_values in value_dict.items():
        steps, values = get_thresholds_points(line_values, thresholds)
        data_size = int(item)
        x.extend([data_size] * len(steps))
        
        # Modify steps if their sum is zero
        # steps = [int(i*500) for i in range(len(steps))] if sum(steps) == 0 else steps
        
        y.extend(steps)
        
        num_colors.extend(values)
        highest_x.append(data_size)
        
        highest_y.append(max(steps))
    
    x = [i/1000 for i in x]
    highest_x = [i/1000 for i in highest_x]
    
    # Set Seaborn style and get color palette
    sns.set_style("whitegrid")
    
    # Create scatter plot
    fig, ax = plt.subplots(
        # figsize=(10, 7)
        )
    num_colors_normalized = [np.searchsorted(thresholds, val) / len(thresholds) for val in num_colors]
    custom_cmap = sns.cubehelix_palette(start=.5, rot=-.75, as_cmap=True)
    sc = ax.scatter(x, y, c=num_colors_normalized, cmap=custom_cmap)
    
    # Connect the highest points using a dashed line with color matching the seaborn palette
    line_color = sns.cubehelix_palette(start=.5, rot=-.75)[3]
    # set first step = 0
    if key == "Test Acc":
        highest_y[0] = 0
    ax.plot(highest_x, highest_y, '--', color=line_color, label="Highest Points")
    # ax.legend()
    if y_max is not None and key == "Train Acc":
        ymin, _ = ax.get_ylim()
        ax.set_ylim(ymin, y_max)
        
    ax.set_xlabel("Data Size (1e3)", fontsize=18)
    ax.set_ylabel("Step", fontsize=18)
    ax.tick_params(axis='y', labelsize=13)
    ax.tick_params(axis='x', labelsize=13)
    
    
    # Add colorbar and label
    cbar = plt.colorbar(sc, ax=ax)
    cbar.ax.tick_params(labelsize=13)  # 设置颜色条刻度标签的字体大小
    # cbar = plt.colorbar(sc, ax=ax, ticks=np.linspace(0, 1, len(thresholds)))
    # cbar.ax.set_yticklabels(thresholds)
    cbar.set_label('Acc', fontsize=18)

    # Save the plot
    if save_file:
        plt.savefig(save_file, dpi=500, bbox_inches="tight")
    else:
        plt.savefig("test.png", dpi=500, bbox_inches="tight")


def read_vaules(files,key=None, clip=6000, sigma=100):
    value_dict = {}
    for f in files:
        log = read_logs(f)
        values = log[key]
        if clip:
            values = values[:clip]
        line = gaussian_filter(values, sigma=sigma)
        numbers = re.findall(r"(\d+)", f)
        if len(numbers) >= 1:
            dir_name = numbers[-1]
        else:
            dir_name = "all_data"
        # dir_name = re.search(r"(\d+)", f).group(0) if re.search(r"(\d+)", f) else "all_data"
        value_dict[dir_name] = line


    # Remove specific keys
    for unwanted_key in ["all_data", "9500", "9000", "8500", "8000", "7500"]:
        if unwanted_key in value_dict:
            value_dict.pop(unwanted_key, None)
    return value_dict


def heatmap(files, save_file=None, key=None, clip=6000, sigma=100):
    # Read values from the provided files
    value_dict = read_vaules(files, key=key, clip=clip, sigma=sigma)

    # Sort and reverse the dictionary keys
    keys_sorted = sorted(value_dict.keys(), reverse=True)
    
    # Create a matrix from the sorted values
    matrix = [value_dict[key] for key in keys_sorted]
    
    # Convert the list to numpy array and sub-sample every 200 columns
    step = 200
    values = np.array(matrix)[:, ::200]

    # Set seaborn style
    sns.set_theme(style="white")
    
    # Initialize the figure
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Create heatmap
    # sns.heatmap(
    #     values, 
    #     cmap="RdBu_r", 
    #     center=0.5, 
    #     linewidths=0.5,
    #     vmin=0, 
    #     vmax=0.9, 
    #     cbar_kws={'label': 'Acc'},
    #     ax=ax
    # )
    # Create heatmap and get the colorbar handle
    cbar_ax = sns.heatmap(
        values, 
        cmap="RdBu_r", 
        center=0.5, 
        linewidths=0.5,
        vmin=0, 
        vmax=0.9, 
        cbar_kws={'label': key},  # 创建颜色条，标记为'Acc'
        ax=ax
    ).collections[0].colorbar
    cbar_ax.set_label(key, size=22) 
    cbar_ax.ax.tick_params(labelsize=17)
    # Set y-tick labels based on the reversed order of keys
    ax.set_yticklabels([int(i)/1000 for i in keys_sorted], rotation=0)
    
    # Set x-ticks and labels
    # step = 200
    original_ticks = [i for i in range(0, clip+1, 1000)]
    tick_positions = [tick/step for tick in original_ticks]
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(original_ticks)

    # Set x and y labels
    ax.set_xlabel("Step", fontsize=22)
    ax.set_ylabel("Data Size (1e3)", fontsize=22)
    ax.tick_params(axis='y', labelsize=17)
    ax.tick_params(axis='x', labelsize=17)
    if key == "Test Acc":
        ax.set_title("Test accuracy across training process", fontsize=24, pad=20)
    else:
        ax.set_title("Train accuracy across training process", fontsize=24, pad=20)
    # Save the figure
    plt.savefig(save_file, dpi=500, bbox_inches="tight")
    plt.close(fig)

def acc_line(files, save_file=None, key_1="Train Loss", key_2="Test Loss", key_3="Train Acc", key_4="Test Acc",
             clip=6000, 
             sigma=100):
    # Read values from the provided files
    logs_dicts = [read_logs(i) for i in files]
    
    def extract_and_smooth(logs, key):
        smoothed_logs = []
        
        for log_set in logs:
            
            # values = [log[key] for log in log_set if key in log]
            smoothed = gaussian_filter(log_set[key], sigma=sigma)
            smoothed_logs.append(smoothed)
            
        return np.array(smoothed_logs)


    # Extract and smooth data for each experiment
    train_losses = extract_and_smooth(logs_dicts, key_1)
    test_losses = extract_and_smooth(logs_dicts, key_2)
    train_accs = extract_and_smooth(logs_dicts, key_3)
    test_accs = extract_and_smooth(logs_dicts, key_4)
    
    # Compute mean and standard deviation for each step
    train_loss_mean = np.mean(train_losses, axis=0)
    train_loss_std = np.std(train_losses, axis=0)
    test_loss_mean = np.mean(test_losses, axis=0)
    test_loss_std = np.std(test_losses, axis=0)
    train_acc_mean = np.mean(train_accs, axis=0)
    train_acc_std = np.std(train_accs, axis=0)
    test_acc_mean = np.mean(test_accs, axis=0)
    test_acc_std = np.std(test_accs, axis=0)
    
    # Plot
    colors = sns.color_palette("tab10")
    fig, ax1 = plt.subplots(figsize=(10, 6))

    # Plot data with fill_between for errors
    ax1.plot(range(len(train_loss_mean)), train_loss_mean, color=colors[0], linestyle='--', label=key_1.split("/")[-1])
    ax1.fill_between(range(len(train_loss_mean)), train_loss_mean - train_loss_std, train_loss_mean + train_loss_std, color=colors[0], alpha=0.2)

    ax2 = ax1.twinx()
    ax2.plot(range(len(train_acc_mean)), train_acc_mean, color=colors[2], label=key_3.split("/")[-1])
    ax2.fill_between(range(len(train_acc_mean)), train_acc_mean - train_acc_std, train_acc_mean + train_acc_std, color=colors[2], alpha=0.2)

    ax2.plot(range(len(test_acc_mean)), test_acc_mean, color=colors[3], label=key_4.split("/")[-1])
    ax2.fill_between(range(len(test_acc_mean)), test_acc_mean - test_acc_std, test_acc_mean + test_acc_std, color=colors[3], alpha=0.2)

    # Legend
    lines = [line for line in ax1.get_lines() + ax2.get_lines()]
    labels = [line.get_label() for line in lines]
    ax1.legend(lines, labels, loc="upper center", bbox_to_anchor=(0.5, 1.08), ncol=4, prop={'size': 13})
    # 修改 x 轴的刻度标签
    locs, _ = plt.xticks()
    plt.xticks(locs[1:-1], [int(i/100000) for i in locs][1:-1])
    
    ax1.set_xlabel("Steps (1e5)", fontsize=15)
    ax1.set_ylabel("Loss",  fontsize=15)
    ax1.tick_params(axis='y', labelsize=15)
    ax1.tick_params(axis='x', labelsize=15)
    ax1.spines['top'].set_visible(False)  

    ax2.set_ylabel("Accuracy", fontsize=15)
    ax2.tick_params(axis='y', labelsize=15)
    ax2.spines['top'].set_visible(False)
    
    
    fig.tight_layout()
    plt.savefig(save_file, dpi=500, bbox_inches="tight")
    
if __name__ == '__main__':
    # training process
    in_path = "model/grokking/modular_v4"
    files = read_subdirs(in_path)
    # key = "Train Loss"
    # save_f = os.path.join("plot/grokking_modular_data_sizes", f"{key}.png")
    # plot_multip_lines(files, save_file=save_f,  key=key, sigma=100)
    
    
    # scatter plot
    # key = "Test Acc"
    key = "Train Acc"
    files.remove('model/grokking/modular_v4/6000_data_grokking_v3/train_log.txt')
    save_f = os.path.join("plot/grokking_modular_data_sizes", f"{key}_scatter_new_bar_new_size.png")
    # set y_max for train acc
    # Test Acc
    # scatter_plot(files, save_file=save_f, key=key, y_max=500)
    
    # heatmap
    # key = "Train Acc"
    # key = "Test Acc"
    # save_f = os.path.join("plot/grokking_modular_data_sizes", f"{key}_heatmap_new_size.png")
    # heatmap(files, save_file=save_f, key=key, clip=2000)
    
    
    # Acc line figure TODO：not work now
    # file_list = ["model/grokking/imdb_1024/imdb_grokking_v1.0", "model/grokking/imdb_1024/imdb_grokking_v1.1",
    #              "model/grokking/imdb_1024/imdb_grokking_v1.2"]
    # file_list = [i+"/train_log.txt" for i in file_list]
    # acc_line(file_list, save_file="test.png", clip=None, sigma=1000)