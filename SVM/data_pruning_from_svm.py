# trained_svm = "./model/svm/svm_model.pkl"

# # load Trained SVM
# with open(trained_svm, "rb") as f:
#     clf = pickle.load(f)
# print("----------- Load SVM Model ---------------")

# # clf 是你的训练好的 SVM 模型
# support_vectors = clf.support_vectors_  # 支持向量
# support_indices = clf.support_  # 支持向量的索引
# support_coefficients = clf.dual_coef_  # 支持向量的系数

# # save indices
# save_supportting_factors = trained_svm.replace("svm_model.pkl", "support_indices.txt")
# with open(save_supportting_factors, "w") as f:
#     for i in support_indices:
#         f.write(str(i) + "\n")

# # # pruning train sert
# data_path = "datasets/natural-instructions-2.8/yesno_task/datatsets/train.json"
# pruning_path = "./datasets/natural-instructions-2.8/yesno_task/datatsets/pruning_set/train.json"
# with open(data_path, "r") as f:
#     data = f.readlines()
    
# pruning_data = np.array(data)[support_indices].tolist()

# with open(pruning_path, "w") as f:
#     for i in pruning_data:
#         f.write(i)
        
import pickle
import numpy as np

def load_svm_model(model_path: str):
    """
    加载 SVM 模型。
    参数:
    - model_path: 存储训练好的 SVM 模型的路径。
    
    返回:
    - 训练好的 SVM 模型。
    """
    with open(model_path, "rb") as f:
        return pickle.load(f)

def save_support_indices(indices: np.array, save_path: str):
    """
    保存支持向量的索引。
    参数:
    - indices: 支持向量的索引。
    - save_path: 保存索引的文件路径。
    """
    with open(save_path, "w") as f:
        for i in indices:
            f.write(str(i) + "\n")

def prune_dataset(data_path: str, indices: np.array, save_path: str):
    """
    基于支持向量的索引修剪数据集。
    参数:
    - data_path: 原始数据集的路径。
    - indices: 支持向量的索引。
    - save_path: 修剪后的数据集的保存路径。
    """
    with open(data_path, "r") as f:
        data = f.readlines()
    
    pruned_data = np.array(data)[indices].tolist()

    with open(save_path, "w") as f:
        for item in pruned_data:
            f.write(item)

if __name__ == "__main__":
    trained_svm_path = "../model/svm-on-support-vector/svm_model_support_vector.pkl"
    data_path = "datasets/natural-instructions-2.8/yesno_task/datasets/train.json"
    pruning_save_path = "./datasets/natural-instructions-2.8/yesno_task/datasets/pruning_set/train.json"

    # 加载训练好的 SVM 模型
    clf = load_svm_model(trained_svm_path)
    print("----------- Load SVM Model ---------------")

    # 获取并保存支持向量的索引
    support_indices = clf.support_
    save_path = trained_svm_path.replace("svm_model.pkl", "support_indices.txt")
    # save_support_indices(support_indices, save_path)
    
    support_vectors = clf.support_vectors_  # 支持向量
    support_indices = clf.support_  # 支持向量的索引
    support_coefficients = clf.dual_coef_  # 支持向量的系数
    print(support_coefficients)
    print("--------------")

    # # 基于支持向量的索引修剪数据集
    # prune_dataset(data_path, support_indices, pruning_save_path)
