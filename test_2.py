import numpy as np

# Create the original array
original_array = np.array([[1, 2, 3], [4, 5, 6]])

# Create a deep copy of the original array
deep_copy_array = original_array.copy()

# Modify the deep copy array
deep_copy_array[0, 0] = 99

# Verify that the original array is unchanged
print("Original Array:")
print(original_array)

# Verify the deep copy array
print("Deep Copy Array:")
print(deep_copy_array)
