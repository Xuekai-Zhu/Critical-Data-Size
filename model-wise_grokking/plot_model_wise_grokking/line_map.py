import matplotlib.pyplot as plt
import seaborn as sns
import os, re
import json
import numpy as np
from tqdm import tqdm
from scipy.ndimage import gaussian_filter
import torch
import statsmodels.api as sm
from matplotlib.ticker import MaxNLocator

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


def line_figure(pruning_levels, accuracies, save_file, top_y=None,):
    # plt.figure(figsize=(10, 6))
    fig = plt.figure()
    plt.plot(pruning_levels, accuracies, marker='o')
    
    if top_y != None:
        plt.axhline(y=top_y, color='gray', linestyle='--')
        # plt.text(0.92*max(pruning_levels), top_y - 0.5, 'SVM Pruning', color='gray', fontsize=12)
        plt.text(min(pruning_levels) - 0.01, top_y - 0.2, 'SVM Pruning', color='black', fontsize=6, fontweight='bold')

    plt.xlabel('Hidden Size')
    plt.ylabel('Avg Acc (%)')
    # plt.title(tittle)
    # plt.xlabel('Hidden Size')
    # plt.ylabel('Average Accuracy (%)')
    plt.title('Average Accuracy vs. Hidden Size')
    plt.grid(True)
    
    # Remove upper and right spines
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    
    plt.savefig(save_file, dpi=500, bbox_inches="tight")


def line_figure_with_reg(pruning_levels, accuracies, save_file, top_y=None):
    # Set the aesthetic style of the plots
    sns.set_style("whitegrid")

    # Prepare data
    x = pruning_levels
    y = accuracies

    # Linear fit
    a, b = np.polyfit(x, y, deg=1)
    y_est = a * np.array(x) + b

    # Add intercept to model for OLS regression
    x_with_intercept = sm.add_constant(x)
    model = sm.OLS(y, x_with_intercept)
    results = model.fit()

    # Get prediction standard errors
    predictions = results.get_prediction(x_with_intercept)
    predictions_summary_frame = predictions.summary_frame()
    y_err_lower = predictions_summary_frame['mean_ci_lower']
    y_err_upper = predictions_summary_frame['mean_ci_upper']

    # Create a color palette
    palette = sns.color_palette("dark:#5A9_r")

    # Create a figure and a set of subplots
    fig, ax = plt.subplots(figsize=(8, 6))
    # ax.set_yscale('log')

    # Plot estimated values and actual values
    ax.plot(x, y_est, '-')  # Estimations with the second color in palette
    ax.scatter(x, y, color=palette[0])  # Actual values with the first color in palette

    # Fill between with a Seaborn color
    ax.fill_between(x, y_err_lower, y_err_upper, alpha=0.2)

    # Add arrow annotation
    ax.annotate('', xy=(x[-1], y_est[-1]), xytext=(x[-2], y_est[-2]),
                arrowprops=dict(facecolor='black', arrowstyle='->', lw=3))

    # Set the labels and title
    ax.set_xlabel('Hidden Size', fontsize=15)
    ax.set_ylabel('Avg Acc', fontsize=15)
    ax.set_title('Average Accuracy vs. Hidden Size')
    ax.yaxis.set_major_locator(MaxNLocator(nbins=2)) 

    # Configure grid and spines
    ax.grid(True, linestyle='--', alpha=0.5)
    sns.despine()  # Remove the top and right spines

    # Save the figure
    plt.savefig(save_file, dpi=500, bbox_inches="tight")


def line_map_for_grokking(in_dir, save_file=None):
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
            test_acc = [i["Test Acc"] for i in single_frac_acc]
            line = gaussian_filter(test_acc, sigma=50)
            all_subdata_acc.append(max(line))
        all_acc.append(all_subdata_acc)
    
    acc = np.array(all_acc) # 16(16 ~ 256 hidden size) x 10 (0.1 ~ 1.0 data frac)
    avg_acc = np.mean(acc, axis=-1)
    hidden_sizes_num = np.arange(16, 272, 16)
    
    # 绘制图表
    # line_figure(hidden_sizes_num, avg_acc, save_file)
    line_figure_with_reg(hidden_sizes_num, avg_acc, save_file)
    
    
if __name__ == '__main__':
    # v2
    base_dir = "model-wise_grokking/model/encoder"
    # heatmap_hidden_size(base_dir)
    
    # heatmap_data_frac(base_dir)
    
    line_map_for_grokking(base_dir, "model-wise_grokking/plot_model_wise_grokking/figures/line_figure_log_scale.png")