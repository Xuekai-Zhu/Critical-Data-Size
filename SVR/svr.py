from sklearn.svm import SVR
import numpy as np
import matplotlib.pyplot as plt


# 设置随机数生成器的种子
np.random.seed(0)

# Create synthetic data
np.random.seed(0)  # for reproducibility
X = np.sort(10 * np.random.rand(100, 1), axis=0)
y = np.sin(X).ravel()
y[::10] += 3 * (0.5 - np.random.rand(10))  # add some noise


# Fit SVR model
svr_rbf = SVR(kernel='rbf', C=1e3, gamma=0.1)
y_rbf = svr_rbf.fit(X, y).predict(X)

# Get the support vectors
support_vectors = svr_rbf.support_vectors_
print(support_vectors)

# Identify the support vectors in the original data
support_vector_indices = svr_rbf.support_
support_vectors_X = X[support_vector_indices]
support_vectors_y = y[support_vector_indices]

# Visualize results with correct support vectors
plt.figure(figsize=(12, 6))
plt.scatter(X, y, color='darkorange', label='Data')
plt.scatter(support_vectors_X, support_vectors_y, facecolors='none', edgecolors='k', s=100, label='Support Vectors')
plt.plot(X, y_rbf, color='navy', lw=2, label='RBF model')
plt.legend()
plt.title('Support Vector Regression with Correct Support Vectors')
plt.xlabel('X')
plt.ylabel('y')
# plt.show()


plt.savefig("./figures/test.png", dpi=500, bbox_inches="tight")
print("!!!!!!!")