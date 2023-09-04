import matplotlib.pyplot as plt
import numpy as np


def line_figure(pruning_levels, accuracies, save_file, top_y=None, tittle='Accuracy vs. Pruning Level'):
    # plt.figure(figsize=(10, 6))
    fig = plt.figure()
    plt.plot(pruning_levels, accuracies, marker='o')
    
    if top_y != None:
        plt.axhline(y=top_y, color='gray', linestyle='--')
        # plt.text(0.92*max(pruning_levels), top_y - 0.5, 'SVM Pruning', color='gray', fontsize=12)
        plt.text(min(pruning_levels) - 0.01, top_y - 0.2, 'SVM Pruning', color='black', fontsize=6, fontweight='bold')

    plt.xlabel('Frac. data kept')
    plt.ylabel('Accuracy (%)')
    plt.title(tittle)
    plt.grid(True)
    
    # Remove upper and right spines
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    
    plt.savefig(save_file, dpi=500, bbox_inches="tight")

if __name__ == '__main__':
    # from pretrain
    # pruning_levels = [0.1, 0.2, 0.3, 
    #             #   0.4, 
    #               0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    # accuracies = [81.87, 84.25, 84.88, 
    #             #   None, 
    #             86.23, 85.79, 86.59, 86.19, 86.24, 87.15]
    # save_file = "pruning_figures/pretrained_opt_pruning.png"
    # top_y = 84.65
    # line_figure(pruning_levels, accuracies, save_file, top_y=top_y, tittle="Based on opt-1.3b")
    
    # from config
    # pruning_levels = [0.1, 0.3, 0.6, 0.8, 1.0]
    # accuracies = [59.56, 55.09, 58.4, 53.93, 56.46]
    
    # save_file = "pruning_figures/From_Scratch_opt_pruning.png"
    # top_y = 54.02
    # line_figure(pruning_levels, accuracies, save_file, top_y=top_y, tittle="Trained From Scratch (1.3b)")
    
    # random pruning on whole train dataset
    pruning_levels = [0.3, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    accuracies = [78.54, 42.91, 71.46, 61.37, 76.07, 56.84, 53.51]
    save_file = "pruning_figures/random_pruning_on_whole_trainset.png"
    top_y = 75.29
    line_figure(pruning_levels, accuracies, save_file, top_y=top_y, tittle="random_pruning_on_whole_train_dataset")
