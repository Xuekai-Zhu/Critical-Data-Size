import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
import pickle
from sklearn.metrics import accuracy_score
import time

def load_data(data_path: str, target_path: str):
    """
    加载特征数据和标签数据。
    参数:
    - data_path: 特征数据的路径。
    - target_path: 标签数据的路径。
    
    返回:
    - features: 特征数据。
    - labels: 标签数据。
    """
    features = np.load(data_path)
    labels = np.load(target_path)
    return features, labels

def train_svm_model(features: np.array, labels: np.array):
    """
    训练 SVM 模型。
    参数:
    - features: 特征数据。
    - labels: 标签数据。
    
    返回:
    - 训练好的 SVM 模型。
    """
    clf = make_pipeline(StandardScaler(), SVC(gamma='auto'))
    clf.fit(features, labels)
    return clf

def save_model(model, save_path: str):
    """
    存储训练好的模型。
    参数:
    - model: 训练好的模型。
    - save_path: 模型的保存路径。
    """
    with open(save_path, "wb") as f:
        pickle.dump(model, f)

def evaluate_model(model, test_features: np.array, test_labels: np.array, save_path: str):
    """
    评估模型的准确性并保存结果。
    参数:
    - model: 训练好的模型。
    - test_features: 测试特征数据。
    - test_labels: 测试标签数据。
    - save_path: 结果的保存路径。
    """
    predictions = model.predict(test_features)
    accuracy = accuracy_score(test_labels, predictions)
    print('Accuracy:', accuracy)
    with open(save_path, "w") as f:
        f.write('Accuracy:{}'.format(accuracy))

if __name__ == "__main__":
    # Load training data
    # train_data_path = "../datasets/yesno_data/vector_data_from_bert/train_pruning_features.npy"
    train_data_path = "../datasets/yesno_data/vector_data_from_bert/train_features.npy"
    train_target_path = train_data_path.replace("features", "labels")
    train_features, train_labels = load_data(train_data_path, train_target_path)

    # Train SVM model
    start_time = time.time()
    print(f"***** Begin Training ******")
    clf = train_svm_model(train_features, train_labels)
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Training took {elapsed_time} seconds")

    # Save the trained model
    save_model_path = "../model/svm/svm_model.pkl"
    save_model(clf, save_model_path)
    print("Model saved!")

    # Load test data
    test_data_path = "../datasets/yesno_data/vector_data_from_bert/train_features.npy"
    test_target_path = test_data_path.replace("features", "labels")
    test_features, test_labels = load_data(test_data_path, test_target_path)

    # Evaluate the model
    result_save_path = "../model/svm/results.eval"
    evaluate_model(clf, test_features, test_labels, result_save_path)







# from sklearn import svm
# import numpy as np
# from sklearn.pipeline import make_pipeline
# from sklearn.preprocessing import StandardScaler
# from sklearn.svm import SVC
# import pickle
# from sklearn.metrics import accuracy_score
# import time
# # from sklearn.model_selection import train_test_split

# # 假设 X 是你的特征数据，y 是你的标签数据
# data_path = "../datasets/yesno_data/vector_data_from_bert/train_pruning_features.npy"
# target_path = data_path.replace("features", "labels")
# train_features = np.load(data_path)
# train_labels = np.load(target_path)

# # 创建SVM模型
# clf = make_pipeline(StandardScaler(), SVC(gamma='auto'))
# clf = svm.SVC()

# # 记录训练开始时间
# start_time = time.time()

# # 训练模型
# clf.fit(train_features, train_labels)

# # 存储模型
# with open("../model/svm/svm_model_support_vector.pkl", "wb") as f:
#     pickle.dump(clf, f) 
# print("Save !!")

# # 记录训练结束时间并计算所花费时间
# end_time = time.time()
# elapsed_time = end_time - start_time

# # 记录训练结束时间并计算所花费时间
# print("--------- Training Complete ----------")
# print(f"Training took {elapsed_time} seconds")

# # load test data
# data_path = "../datasets/yesno_data/vector_data_from_bert/train_features.npy"
# target_path = data_path.replace("features", "labels")
# test_features = np.load(data_path)
# test_labels = np.load(target_path)

# # 测试模型
# predictions = clf.predict(test_features)
# accuracy = accuracy_score(test_labels, predictions)
# print('Accuracy:', accuracy)
# with open("../model/svm/results.eval", "w") as f:
#     f.write('Accuracy:{}'.format(accuracy))


