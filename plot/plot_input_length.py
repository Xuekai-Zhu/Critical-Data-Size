import seaborn as sns
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

def plot_histplot(in_csv, key=None, save_file=None, xlabel=None):
    # 设置Seaborn的风格和调色板
    sns.set_theme(style="darkgrid")
    
    in_data = pd.read_csv(in_csv)
    # print(in_data)
    
    sns.displot(in_data[key], kde=True, bins=20)
    if xlabel is not None:
        plt.xlabel(xlabel)
    plt.show()
    plt.savefig(save_file, dpi=300, 
                # bbox_inches="tight"
                )
    
if __name__ == '__main__':
    # in_file = "plots/code_length.csv"
    # train
    # in_file = "./statics_files/train_input_length.csv"
    # plot_histplot(in_file, key="inputs_length", save_file="./figures/train_input_lengths.png", xlabel="input Length")
    
    # vaild
    in_file = "./statics_files/vaild_input_length.csv"
    plot_histplot(in_file, key="inputs_length", save_file="./figures/vaild_input_lengths.png", xlabel="input Length")
    
    # test
    in_file = "./statics_files/test_input_length.csv"
    plot_histplot(in_file, key="inputs_length", save_file="./figures/test_input_lengths.png", xlabel="input Length")