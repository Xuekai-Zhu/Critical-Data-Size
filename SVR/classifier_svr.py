
# Import required libraries
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVR
import os

# Create more clearly separated synthetic data for the first "class"
np.random.seed(0)  # for reproducibility
X1_adjusted = np.sort(5 * np.random.rand(100, 1), axis=0)
y1_adjusted = np.sin(X1_adjusted).ravel() + 1  # Shifted upward but closer

# Fit SVR model for the first "class"
svr_rbf1 = SVR(kernel='rbf', C=1e3, gamma=0.1)
y_rbf1_adjusted = svr_rbf1.fit(X1_adjusted, y1_adjusted).predict(X1_adjusted)

# Get the support vectors for the first "class"
support_vectors_X1_adjusted = X1_adjusted[svr_rbf1.support_]
support_vectors_y1_adjusted = y1_adjusted[svr_rbf1.support_]

# Create more clearly separated synthetic data for the second "class"
np.random.seed(42)  # for reproducibility
X2_adjusted = np.sort(5 * np.random.rand(100, 1), axis=0)
y2_adjusted = np.cos(X2_adjusted).ravel() - 1  # Shifted downward but closer

# Fit SVR model for the second "class"
svr_rbf2 = SVR(kernel='rbf', C=1e3, gamma=0.1)
y_rbf2_adjusted = svr_rbf2.fit(X2_adjusted, y2_adjusted).predict(X2_adjusted)

# Get the support vectors for the second "class"
support_vectors_X2_adjusted = X2_adjusted[svr_rbf2.support_]
support_vectors_y2_adjusted = y2_adjusted[svr_rbf2.support_]

# Plot with reduced whitespace and only the left and bottom frames
plt.figure(figsize=(14, 8))

# Plot for Adjusted Class 1
plt.scatter(X1_adjusted, y1_adjusted, marker='o', color='darkorange', label='Class 1 Data', zorder=1)
plt.scatter(support_vectors_X1_adjusted, support_vectors_y1_adjusted, s=200, facecolors='none', edgecolors='k', linewidth=1.5, label='Class 1 Support Vectors', zorder=2)
plt.plot(X1_adjusted, y_rbf1_adjusted, color='navy', lw=3, label='Class 1 RBF model', zorder=0)

# Plot for Adjusted Class 2
plt.scatter(X2_adjusted, y2_adjusted, marker='o', color='green', label='Class 2 Data', zorder=1)
plt.scatter(support_vectors_X2_adjusted, support_vectors_y2_adjusted, s=200, facecolors='none', edgecolors='magenta', linewidth=1.5, label='Class 2 Support Vectors', zorder=2)
plt.plot(X2_adjusted, y_rbf2_adjusted, color='red', lw=3, label='Class 2 RBF model', zorder=0)

# # Adjust aesthetics
# plt.gca().spines['right'].set_visible(False)
# plt.gca().spines['top'].set_visible(False)

plt.legend()
plt.title('Classification using Support Vector Regression')
plt.xlabel('X')
plt.ylabel('y')

# Save the figure
save_directory = "SVR/figures"
if not os.path.exists(save_directory):
    os.makedirs(save_directory)
    
adjusted_save_path = os.path.join(save_directory, "svr_classification_adjusted.png")
plt.tight_layout()
plt.savefig(adjusted_save_path, dpi=500, bbox_inches="tight")
