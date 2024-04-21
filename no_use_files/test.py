import torch
import matplotlib.pyplot as plt
import numpy as np

# 生成100x100的模拟图像
blur = np.random.randn(100,100) 

# 设置上采样参数
plot_scale = 2

# 进行上采样
train_large = torch.nn.functional.interpolate(torch.tensor(blur).unsqueeze(dim=0).unsqueeze(dim=0), 
                                             scale_factor=(plot_scale,plot_scale), mode='bilinear')[0,0].detach().numpy()

# 绘图
fig, axs = plt.subplots(1, 2)
axs[0].imshow(blur)
axs[1].imshow(train_large)
plt.savefig("test.png", dpi=500, bbox_inches="tight")