import random
import pandas as pd
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt


def analyze_data_points(npz_file_path):
    # Load the data from the provided npz file
    multipliers_data_1 = np.load(npz_file_path[0])
    multipliers_data_2 = np.load(npz_file_path[1])

    # Initialize lists to store data from the first and second classes
    first_class_sample = []
    second_class_sample = []

    # Randomly sample keys from the multipliers_data
    # sample_keys = random.sample(list(multipliers_data.keys()), sample_size)

    # Iterate through the sampled keys and collect the relevant data points
    # first_class = []
    # second_class = []
    for i, index in enumerate(tqdm(multipliers_data_1.keys())):
        
        mul = multipliers_data_1[index]
        first_class_sample.append(mul[0,0]) 
        second_class_sample.append(mul[0,1]) 
        
    for i, index in enumerate(tqdm(multipliers_data_2.keys())):
        
        mul = multipliers_data_2[index]
        first_class_sample.append(mul[0,0]) 
        second_class_sample.append(mul[0,1]) 
    
    # for index in sample_keys:
    #     mul = multipliers_data[index]
    #     first_class_sample.append(mul[0, 0])
    #     second_class_sample.append(mul[0, 1])
        
    return first_class_sample, second_class_sample

def save_threshold_percentages(first_class_sample, second_class_sample, thresholds, csv_file_path):
    # Calculate the percentage of data points close to 0 for each threshold
    percentages_first_class = []
    percentages_second_class = []

    for threshold in thresholds:
        close_to_zero_first_class = [val for val in first_class_sample if abs(val) < threshold]
        close_to_zero_second_class = [val for val in second_class_sample if abs(val) < threshold]
    
        percentages_first_class.append((len(close_to_zero_first_class) / len(first_class_sample)) * 100)
        percentages_second_class.append((len(close_to_zero_second_class) / len(second_class_sample)) * 100)
        
    # Create a DataFrame to store the results
    df = pd.DataFrame({
        'Threshold': thresholds,
        'Percentage_First_Class': percentages_first_class,
        'Percentage_Second_Class': percentages_second_class
    })

    # Save the DataFrame to a CSV file
    df.to_csv(csv_file_path, index=False)
    
    
# def analyze_quantiles(first_class_sample, second_class_sample, quantiles, csv_file_path):
#     # Calculate the quantiles for the first and second class
#     quantiles_first_class = np.percentile(first_class_sample, quantiles)
#     quantiles_second_class = np.percentile(second_class_sample, quantiles)
    
#     # Create a DataFrame to store the results
#     df_quantiles = pd.DataFrame({
#         'Quantile_Percentage': quantiles,
#         'Quantile_First_Class': quantiles_first_class,
#         'Quantile_Second_Class': quantiles_second_class
#     })

#     # Save the DataFrame to a CSV file
#     df_quantiles.to_csv(csv_file_path, index=False)

def analyze_zero_neighborhood(first_class_sample, second_class_sample, percentages, csv_file_path):
    # Sort the data by absolute value
    sorted_first_class = sorted(first_class_sample, key=abs)
    sorted_second_class = sorted(second_class_sample, key=abs)
    
    # Initialize lists to store the ranges
    ranges_first_class = []
    ranges_second_class = []
    
    # Initialize lists to store the ranges
    actual_ranges_first_class = []
    actual_ranges_second_class = []

    
    # Calculate the ranges for specified percentages
    for percentage in percentages:
        idx = int(len(sorted_first_class) * (percentage / 100))
        min_value_first_class = min(sorted_first_class[:idx+1])
        max_value_first_class = max(sorted_first_class[:idx+1])
        actual_ranges_first_class.append((min_value_first_class, max_value_first_class))
        
        idx = int(len(sorted_second_class) * (percentage / 100))
        min_value_second_class = min(sorted_second_class[:idx+1])
        max_value_second_class = max(sorted_second_class[:idx+1])
        actual_ranges_second_class.append((min_value_second_class, max_value_second_class))
    

    
    # Create a DataFrame to store the results
    df_actual_ranges = pd.DataFrame({
        'Percentage': percentages,
        'Actual_Range_First_Class': actual_ranges_first_class,
        'Actual_Range_Second_Class': actual_ranges_second_class
    })

    # Save the DataFrame to a CSV file
    df_actual_ranges.to_csv(csv_file_path, index=False)

def plot_scatter(samples, save_file):
    array1_expanded = np.expand_dims(samples[0], axis=1)
    array2_expanded = np.expand_dims(samples[1], axis=1)
    all_data = np.concatenate((array1_expanded, array2_expanded), axis=1)

    # all_data = np.concatenate((samples), axis=-1)
    # Plot the scatter plot
    plt.figure(figsize=(10, 8))
    plt.scatter(all_data[:, 0], all_data[:, 1], c='blue', marker='o', s=10)
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.title('Scatter Plot of multiplier')
    plt.grid(True)
    plt.savefig(save_file, dpi=500, bbox_inches="tight")

if __name__ == '__main__':

    # Example usage
    npz_file_path_1 = 'model/bert-base-uncased-unpruning/lagrange_multiplier/project_final_multiplier_0.5.npz'  # Replace with the actual file path
    npz_file_path_2 = 'model/bert-base-uncased-unpruning/lagrange_multiplier/project_final_multiplier.npz'  # Replace with the actual file path
    csv_file_path = 'model/bert-base-uncased-unpruning/lagrange_multiplier/figures/threshold_percentages_example.csv'  # Replace with your desired output CSV file path
    csv_quantile_file_path = 'model/bert-base-uncased-unpruning/lagrange_multiplier/figures/zero_neighborhood_values.csv'  # Replace with your desired output CSV file path
    # sample_size = 5000  # You can adjust this number
    thresholds = [0.01, 0.05, 0.1, 0.5]  # You can adjust this list

    first_class_sample, second_class_sample = analyze_data_points([npz_file_path_1, npz_file_path_2])
    # save_threshold_percentages(first_class_sample, second_class_sample, thresholds, csv_file_path)


    percentages = [1, 5, 10, 20, 30]  # You can adjust this list
    # Run the quantile analysis function
    # analyze_zero_neighborhood(first_class_sample, second_class_sample, percentages, csv_quantile_file_path)

    # plot 
    png_file = "model/bert-base-uncased-unpruning/lagrange_multiplier/figures/all_multiplier"
    plot_scatter([first_class_sample, second_class_sample], png_file)


