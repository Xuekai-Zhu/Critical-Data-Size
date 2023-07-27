from sklearn import svm
import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
import pickle
from sklearn.metrics import accuracy_score
import time
# from sklearn.model_selection import train_test_split

# 假设 X 是你的特征数据，y 是你的标签数据
data_path = "./datasets/natural-instructions-2.8/yesno_task/datatsets/vector_data_from_bert/train_features.npy"
target_path = data_path.replace("features", "labels")
train_features = np.load(data_path)#[:100, :]
train_labels = np.load(target_path)#[:100]

# 创建SVM模型
clf = make_pipeline(StandardScaler(), SVC(gamma='auto'))
clf = svm.SVC()

# 记录训练开始时间
start_time = time.time()

# 训练模型
clf.fit(train_features, train_labels)

# 记录训练结束时间并计算所花费时间
end_time = time.time()
elapsed_time = end_time - start_time

# 记录训练结束时间并计算所花费时间
print("--------- Training Complete ----------")
print(f"Training took {elapsed_time} seconds")

# load test data
data_path = "./datasets/natural-instructions-2.8/yesno_task/datatsets/vector_data_from_bert/test_features.npy"
target_path = data_path.replace("features", "labels")
test_features = np.load(data_path)
test_labels = np.load(target_path)

# 测试模型
predictions = clf.predict(test_features)
accuracy = accuracy_score(test_labels, predictions)
print('Accuracy:', accuracy)
with open("./model/svm/results.eval", "w") as f:
    f.write('Accuracy:{}'.format(accuracy))

# 存储模型
with open("./model/svm/svm_model.pkl", "wb") as f:
    pickle.dump(clf, f) 
print("Save !!")
