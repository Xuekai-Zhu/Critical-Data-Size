from sklearn.svm import SVR
import numpy as np
import matplotlib.pyplot as plt
import os

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
# print(support_vectors)

# Identify the support vectors in the original data
support_vector_indices = svr_rbf.support_
support_vectors_X = X[support_vector_indices]
support_vectors_y = y[support_vector_indices]

# Get dual coefficients (Lagrange multipliers for the support vectors)
dual_coef = svr_rbf.dual_coef_
print(dual_coef)

# Calculate Lagrange multipliers for all data points
# Note: For non-support vectors, the Lagrange multipliers are essentially zero.
all_dual_coef = np.zeros_like(y)
all_dual_coef[support_vector_indices] = dual_coef

# Plot
plt.figure(figsize=(12, 6))

# Visualize results with correct support vectors
plt.subplot(1, 2, 1)
plt.scatter(X, y, color='darkorange', label='Data')
plt.scatter(support_vectors_X, support_vectors_y, facecolors='none', edgecolors='k', s=100, label='Support Vectors')
plt.plot(X, y_rbf, color='navy', lw=2, label='RBF model')
plt.legend()
plt.title('Support Vector Regression with Correct Support Vectors')
plt.xlabel('X')
plt.ylabel('y')
# plt.show()

# Plot Lagrange multipliers for all data points
plt.subplot(1, 2, 2)
plt.scatter(X, all_dual_coef, color='blue', s=10, label='Non-support vectors')
plt.scatter(support_vectors_X, dual_coef, color='purple', s=30, label='Support vectors')
plt.title('Lagrange Multipliers for All Data Points')
plt.xlabel('X')
plt.ylabel('Lagrange Multiplier')
plt.legend()

# Save the modified figure
save_directory = "SVR/figures"
plt.tight_layout()
modified_save_path = os.path.join(save_directory, "svr_with_all_lagrange_multipliers.png")
plt.savefig(modified_save_path, dpi=500, bbox_inches="tight")

print("!!!!!!!")