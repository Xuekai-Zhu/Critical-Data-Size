import matplotlib.pyplot as plt
import seaborn as sns
import os, re
import json
import numpy as np
from tqdm import tqdm
from scipy.ndimage import gaussian_filter

# Thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
Thresholds = [0.5, 0.55, 0.575, 0.6, 0.625, 0.65, 0.675, 0.7, 0.725, 0.75, 0.8, 0.85]

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


def get_thresholds_points(sequence, thresholds):
    # thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    # Finding the indices of values just after each threshold
    # indices = [np.argmax(sequence >= threshold) for threshold in thresholds]

    indices = []
    for threshold in thresholds:
        if np.any(sequence >= threshold):
            index = np.argmax(sequence >= threshold)
            indices.append(index)
        # else:
        #     indices.append(0)


    # Extracting the values corresponding to the found indices
    # if sequence == 0:
    #     values = np.zeros((len(indices)))
    # else:
    values = sequence[indices]
    
    return indices, values

def scatter_plot(x,y, highest_x, highest_y, num_colors,
                 save_file=None, key=None, y_max=None):

    
    # Set Seaborn style and get color palette
    sns.set_style("whitegrid")
    
    # Create scatter plot
    fig, ax = plt.subplots()
    num_colors_normalized = [np.searchsorted(Thresholds, val) / len(Thresholds) for val in num_colors]
    custom_cmap = sns.cubehelix_palette(start=.5, rot=-.75, as_cmap=True)
    sc = ax.scatter(x, y, c=num_colors_normalized, cmap=custom_cmap)
    
    # Connect the highest points using a dashed line with color matching the seaborn palette
    line_color = sns.cubehelix_palette(start=.5, rot=-.75)[3]

    ax.plot(highest_x, highest_y, '--', color=line_color, label="Highest Points")
    ax.legend()

    ax.set_xlabel("Data Size (1e3)", fontsize=12)
    ax.set_ylabel("Step", fontsize=12)
    
    
    # Add colorbar and label
    cbar = plt.colorbar(sc, ax=ax, ticks=np.linspace(0, 1, len(Thresholds)))
    cbar.ax.set_yticklabels(Thresholds)  # Set colorbar labels to actual thresholds values
    # cbar = plt.colorbar(sc, ax=ax)
    cbar.set_label('Acc')

    # Save the plot
    if save_file:
        plt.savefig(save_file, dpi=500, bbox_inches="tight")
    else:
        plt.savefig("test.png", dpi=500, bbox_inches="tight")


def obtain_data(input_data, data_size=None):
    # thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    x, y, num_colors = [], [], []
    highest_x, highest_y = [], []
    for item, line_values in enumerate(input_data):
        steps, values = get_thresholds_points(line_values, Thresholds)
        data_size_now = data_size[item]
        x.extend([data_size_now] * len(steps))
        
        # Modify steps if their sum is zero
        # steps = [int(i*500) for i in range(len(steps))] if sum(steps) == 0 else steps
        
        y.extend(steps)
        
        num_colors.extend(values)
        highest_x.append(data_size_now)
        
        highest_y.append(max(steps))
        
    return x, y, highest_x, highest_y, num_colors

def main(in_dir):
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
            
        x, y, highest_x, highest_y, num_colors = obtain_data(all_subdata_acc, data_size=data_size)
        
        save_dir = os.path.join("model-wise_grokking/plot_model_wise_grokking/figures", f"{hidden_size_i}")
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
            
        save_png = os.path.join(save_dir, f"{data_frac}.png")
        
        scatter_plot(x, y, highest_x, highest_y, num_colors, save_file=save_png)
    
    
if __name__ == '__main__':
    base_dir = "model-wise_grokking/model/encoder"
    main(base_dir)