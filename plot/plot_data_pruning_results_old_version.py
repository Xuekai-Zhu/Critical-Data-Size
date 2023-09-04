import seaborn as sns
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt


def line_plot(data, save_file, top_y=None):
    # sns.set_theme(style="whitegrid")
    # sns.set_theme(style="white")
    # sns.set_theme(style="darkgrid")
    sns.set_theme(style="whitegrid", font_scale=1.1)

    data = pd.DataFrame(data)
    g = sns.relplot(
                x="Frac. data kept",
                y="Solve Rate (%)",
                style="pruning",
                hue="pruning",
                data=data, 
                kind="line",
                # kind="reg",
                markers=True, 
                dashes=False,
                # palette="tab10", linewidth=2.5
                )
    if top_y is not None:
        plt.axhline(y=top_y, color='gray', linestyle='--', label='no_pruning')
    # plt.legend()
    # 获取FacetGrid对象的轴，并设置x轴的刻度和标签
    ax = g.axes[0, 0]
    ax.set_xticks(data["Frac. data kept"].unique())
    ax.set_xticklabels(data["Frac. data kept"].unique())
    
    plt.savefig(save_file, dpi=500, bbox_inches="tight")
    
    
def line_correct_plot(data, save_file, top_y=None):
    sns.set_theme(style="whitegrid", font_scale=1.1)
    # sns.set_theme(style="white")
    # sns.set_theme(style="darkgrid")

    data = pd.DataFrame(data)
    g = sns.relplot(
                x="Frac. data kept",
                y="Grammatically Correct Rate (%)",
                style="Model Scale",
                hue="Model Scale",
                data=data, 
                kind="line",
                # kind="reg",
                markers=True, 
                dashes=False,
                # palette="tab10", linewidth=2.5
                )
    
    plt.axhline(y=top_y, color='gray', linestyle='--')
    # plt.legend()
    # 获取FacetGrid对象的轴，并设置x轴的刻度和标签
    ax = g.axes[0, 0]
    ax.set_xticks(data["Frac. data kept"].unique())
    ax.set_xticklabels(data["Frac. data kept"].unique())
    plt.legend(loc='lower right')
    plt.savefig(save_file, dpi=500, bbox_inches="tight")
    
    
    
if __name__ == '__main__':
    
    # samll model data

    data = {
        "pruning": [
                    # "no_pruning", "no_pruning", "no_pruning", "no_pruning", "no_pruning", "no_pruning", "no_pruning", "no_pruning", "no_pruning","no_pruning",
                    # "compressed_rate", "compressed_rate", "compressed_rate", "compressed_rate", "compressed_rate", "compressed_rate", "compressed_rate", "compressed_rate" ,"compressed_rate", "compressed_rate", 
                    "faithful", "faithful", "faithful", "faithful", "faithful", "faithful", "faithful", "faithful", "faithful", "faithful",
                    "GraNd", "GraNd", "GraNd", "GraNd", "GraNd", "GraNd", "GraNd", "GraNd", "GraNd", "GraNd",
                    "entropy", "entropy", "entropy", "entropy", "entropy", "entropy", "entropy", "entropy", "entropy", "entropy",
                    ],
        "Solve Rate (%)": [ 
                        # 30.64, 30.64, 30.64, 30.64, 30.64, 30.64, 30.64, 30.64, 30.64, 30.64,
                        # 34.37, 34.44, 33.91, 32.54, 30.26, 27.30, 25.55, 21.06, 20.6, 10.11,
                        34.37, 34.60, 32.39, 31.25, 28.28, 26.99, 25.32, 22.12, 17.79, 11.55,
                        34.37, 33.30, 30.79, 29.42, 28.74, 24.79, 24.48, 21.82, 18.93, 13.23,
                        34.37, 33.46, 30.57, 30.41, 29.35, 26.53, 24.33, 19.46, 16.19, 7.98,
                        ],
        
        "Frac. data kept": [
                            # 1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 
                            # 1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 
                            1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1,
                            1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1,
                            1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1,
                            ],
    }
    
    # line_plot(data, "plots/paper_figure/data_pruning_figure/base_model_pruning_line_plot.png", top_y=34.37)



    
    # base model data 
    data = {}
    data["Frac. data kept"] = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

    data["Acc (%)"] = [81.87, 84.25, 84.88, None, 86.23, 85.79, 86.59, 86.19, 86.24]

    line_correct_plot(data, "plots/paper_figure/gramatica_correct/entropy_Grammatically_Correct_line_plot.png", top_y=89.9)