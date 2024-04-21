import numpy as np
from tqdm import tqdm
import json
import os

def load_json_file(in_file, return_dict=False):
    with open(in_file, "r") as f:
        data = f.readlines()
    if return_dict:
        all_data = [json.loads(i) for i in data]
        return all_data
    else:
        return data
    


def analyze_data_points(npz_file_path):
    # Load the data from the provided npz file
    multipliers_data_1 = np.load(npz_file_path[0])
    multipliers_data_2 = np.load(npz_file_path[1])

    # Initialize lists to store data from the first and second classes
    first_class_sample = []
    second_class_sample = []

    for i, index in enumerate(tqdm(multipliers_data_1.keys())):
        
        mul = multipliers_data_1[index]
        first_class_sample.append(mul[0,0]) 
        second_class_sample.append(mul[0,1]) 
        
    for i, index in enumerate(tqdm(multipliers_data_2.keys())):
        
        mul = multipliers_data_2[index]
        first_class_sample.append(mul[0,0]) 
        second_class_sample.append(mul[0,1]) 
    
    return first_class_sample, second_class_sample



def sorted_data(multiplier, index_save_path):
    # Load all_data
    # all_data = load_json_file(data, return_dict=False)
    
    # sorted
    sorted_indices = np.argsort(np.abs(multiplier), axis=None)  # Sort by absolute value and get indices
    with open(index_save_path, "w") as f:
        for i in sorted_indices:
            f.write(str(i) + "\n")
    
    
    # # Initialize lists to store the deleted indices for each percentage
    # indices_to_delete = []

    # # Calculate the indices to delete for each percentage
    # for percentage in percentages:
    #     idx = int(len(sorted_indices) * (percentage / 100))
    #     indices_to_delete.append(sorted_indices[:idx+1])

    # # Delete the data points
    # for indices in indices_to_delete:
    #     filtered_data = np.delete(all_data, indices)
    #     deleted_indices = indices

def data_pruning(index_file, percentages, orignal_data_path, save_path, multip_class=False):
    # Load all_data
    all_data = load_json_file(orignal_data_path, return_dict=False)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
        
    # # Calculate the indices to delete for each percentage
    if multip_class:
        with open(index_file[0], "r") as f:
            indexs_1 = f.readlines()
            sorted_indices_1 = [int(i) for i in indexs_1]
        with open(index_file[1], "r") as f:
            indexs_2 = f.readlines()
            sorted_indices_2 = [int(i) for i in indexs_2]
        # # Initialize lists to store the deleted indices for each percentage
        indices_to_delete = []
        for percentage in tqdm(percentages):
            idx_1 = int(len(sorted_indices_1) * (percentage / 100))
            idx_2 = int(len(sorted_indices_2) * (percentage / 100))
            intersection = list(set(sorted_indices_1[:idx_1+1]) & set(sorted_indices_2[:idx_2+1]))
            
            indices_to_delete.append(intersection)
            pruning_data = np.delete(all_data, intersection)
            
            save_file = os.path.join(save_path, "{}_SVR_pruning_train.json".format(percentage))
            with open(save_file, "w") as f_out:
                for i in pruning_data:
                    f_out.write(i)
            
            
    else:
        with open(index_file, "r") as f:
            indexs = f.readlines()
            sorted_indices = [int(i) for i in indexs]
        
        # # Initialize lists to store the deleted indices for each percentage
        indices_to_delete = []
        for percentage in tqdm(percentages):
            idx = int(len(sorted_indices) * (percentage / 100))
            indices_to_delete.append(sorted_indices[:idx+1])
            pruning_data = np.delete(all_data, sorted_indices[:idx+1])
            
            save_file = os.path.join(save_path, "{}_SVR_pruning_train.json".format(percentage))
            with open(save_file, "w") as f_out:
                for i in pruning_data:
                    f_out.write(i)



if __name__ == '__main__':
    
    # Example usage
    # multiplier files
    npz_file_path_1 = 'model/bert-base-uncased/lagrange_multiplier/project_final_multiplier_0.5.npz'  # Replace with the actual file path
    npz_file_path_2 = 'model/bert-base-uncased/lagrange_multiplier/project_final_multiplier.npz'  # Replace with the actual file path
    # first_class_sample, second_class_sample = analyze_data_points([npz_file_path_1, npz_file_path_2])

    # sorted data
    index_save_path_1 = 'model/bert-base-uncased-unpruning/lagrange_multiplier/index_files/class_1.txt'
    index_save_path_2 = 'model/bert-base-uncased-unpruning/lagrange_multiplier/index_files/class_2.txt'
    # sorted_data(first_class_sample, index_save_path_1)
    # sorted_data(second_class_sample, index_save_path_2)
    
    # filter data
    orignal_data_path = "datasets/yesno_task/datatsets/train.json"
    percentages_to_delete = [1, 5, 10, 20, 30]
    filtered_data_save_dir = "datasets/yesno_task/SVR_pruning"
    pruning_first_class = os.path.join(filtered_data_save_dir, "first_class")
    pruning_second_class = os.path.join(filtered_data_save_dir, "second_class")
    # data_pruning(index_save_path_1, percentages_to_delete, orignal_data_path, pruning_first_class)
    # data_pruning(index_save_path_2, percentages_to_delete, orignal_data_path, pruning_second_class)
    
    # multip calss
    percentages_to_delete = [30]
    filtered_data_save_dir = "datasets/yesno_task/SVR_pruning/Multiple_class"
    data_pruning([index_save_path_1, index_save_path_2], percentages_to_delete, orignal_data_path, filtered_data_save_dir, multip_class=True)
    
    


