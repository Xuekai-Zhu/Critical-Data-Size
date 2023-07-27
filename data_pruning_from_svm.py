import pickle
import numpy as np
from sklearn.metrics import accuracy_score

trained_svm = "./model/svm/svm_model.pkl"

# load Trained SVM
with open(trained_svm, "rb") as f:
    clf = pickle.load(f)
print("----------- Load SVM Model ---------------")

# clf 是你的训练好的 SVM 模型
support_vectors = clf.support_vectors_  # 支持向量
support_indices = clf.support_  # 支持向量的索引
support_coefficients = clf.dual_coef_  # 支持向量的系数

# save indices
# save_supportting_factors = trained_svm.replace("svm_model.pkl", "support_indices.txt")
# with open(save_supportting_factors, "w") as f:
#     for i in support_indices:
#         f.write(str(i) + "\n")

# pruning train sert
data_path = "datasets/natural-instructions-2.8/yesno_task/datatsets/train.json"
pruning_path = "./datasets/natural-instructions-2.8/yesno_task/datatsets/pruning_set/train.json"
with open(data_path, "r") as f:
    data = f.readlines()
    
pruning_data = np.array(data)[support_indices].tolist()

with open(pruning_path, "w") as f:
    for i in pruning_data:
        f.write(i)
        
