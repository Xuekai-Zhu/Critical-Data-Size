from transformers import AutoModel, AutoConfig
import torch
import re
from datasets import load_dataset

import matplotlib.pyplot as plt
import numpy as np

pruning_levels = [0.1, 0.2, 0.3, 
                #   0.4, 
                  0.5, 0.6, 0.7, 0.8, 0.9]
accuracies = [81.87, 84.25, 84.88, 
            #   None, 
              86.23, 85.79, 86.59, 86.19, 86.24]

# Convert None values to np.nan
# accuracies = [acc if acc is not None else np.nan for acc in accuracies]

plt.figure(figsize=(10, 6))
plt.plot(pruning_levels, accuracies, marker='o')
plt.axhline(y=top_y, color='gray', linestyle='--')
plt.text(0.92, top_y - 0.5, f'{top_y:.2f}', color='gray', fontsize=12)



plt.xlabel('Frac. data kept')
plt.ylabel('Accuracy (%)')
plt.title('Accuracy vs. Pruning Level')
plt.grid(True)
plt.savefig("test.png", dpi=500, bbox_inches="tight")






