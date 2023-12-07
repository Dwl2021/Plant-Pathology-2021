#!/usr/bin/env python3

import pandas as pd
import matplotlib.pyplot as plt

# 用于存储数据的文件路径
file_paths = {
	"ArcFaceLoss": "CrossViT_ArcFaceLoss.txt",
	"BCELoss": "CrossViT_BCELoss.txt",
	"FocalLoss": "CrossViT_FocalLoss.txt"
}

# 读取数据并转换为数值类型
accuracy_data = {}
for loss_function, file_path in file_paths.items():
	with open(file_path, 'r') as file:
		accuracy_data[loss_function] = [float(line.strip()) for line in file.readlines()]
		
# 创建DataFrame
df_accuracy = pd.DataFrame(accuracy_data)

# 绘制图表
plt.figure(figsize=(8, 4))

# 为每种损失函数指定不同的颜色
colors = {
	"ArcFaceLoss": "#377eb8",  # A shade of blue
	"BCELoss": "#4daf4a",      # A shade of green
	"FocalLoss": "#e41a1c"     # A shade of red
}

for loss_function, color in colors.items():
	plt.plot(df_accuracy.index, df_accuracy[loss_function], label=loss_function, color=color, linewidth=2)
	
	
plt.title('Model Validation Accuracy Over Time by Loss Function', fontsize=14)
plt.xlabel('Epochs', fontsize=12)
plt.ylabel('Accuracy', fontsize=12)
plt.legend(fontsize=12)
plt.grid(True)
plt.show()
