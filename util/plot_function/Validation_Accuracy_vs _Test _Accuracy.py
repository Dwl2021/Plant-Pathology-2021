import matplotlib.pyplot as plt

# 设置英文字体为Times New Roman
plt.rcParams['font.family'] = 'Times New Roman'

# 设置所有文本的默认字体大小
plt.rcParams['font.size'] = 14

# 模型名称
models = ['ResNet50', 'ResNet101', 'SEResNet50', 'SEResNet101', 'SEResNet34', 'SG_ViT', 'CrossViT', 'Deit']

# 验证准确度
valid_acc = [0.713, 0.722, 0.782, 0.758, 0.79, 0.833, 0.85, 0.825]

# 测试准确度
test_acc = [0.77, 0.765, 0.782, 0.768, 0.738, 0.825, 0.803, 0.817]

# Nature 风格的颜色
nature_colors = ['#FF7F0E', '#2CA02C', '#1F77B4', '#D62728', '#9467BD', '#8C564B', '#E377C2', '#7F7F7F']

# 创建一个二维散点图
plt.figure(figsize=(10, 6))

# 绘制每个模型的点，使用 Nature 风格的颜色
for i in range(len(models)):
    plt.scatter(valid_acc[i], test_acc[i], label=models[i], color=nature_colors[i], s=100, alpha=0.7)
    # 在点的旁边添加模型名称，放大字体
    plt.annotate(models[i], (valid_acc[i], test_acc[i]), textcoords="offset points", xytext=(5,0), ha='left', va='center', fontsize=12)

# 设置图表标题和标签，并放大字体
plt.title('Validation Accuracy vs Test Accuracy', fontsize=18)
plt.xlabel('Validation Accuracy', fontsize=16)
plt.ylabel('Test Accuracy', fontsize=16)

# 设置网格线为虚线
plt.grid(True, linestyle='--')

# 显示图表
plt.show()
