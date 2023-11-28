import matplotlib.pyplot as plt
import numpy as np

# 模型和指标数据
models = ['SEResNet50', 'SEResNet101', 'SEResNet34']
metrics = ['Precision', 'Recall', 'F1 Score', 'Valid Accuracy', 'Test Accuracy']

# 每个模型对应的指标值
model_data = {
    'SEResNet50': [0.8310155351283467, 0.807153965785381, 0.8118497513440034, 0.7816666960716248, 0.7816666960716248],
    'SEResNet101': [0.8320537767958381, 0.7931570762052877, 0.801165060576628, 0.7583333253860474, 0.7683333158493042],
    'SEResNet34': [0.8450114523310825, 0.8164852255054432, 0.8227255409191564, 0.7900000214576721, 0.7383333444595337]
}
marker_list = ['o','s','^']
line_styles =['-','-.','--']

# 绘制折线图
plt.figure(figsize=(6, 4))

for i,model in enumerate(models):
    plt.plot(metrics, model_data[model], label=model, marker=marker_list[i],markersize=10,linewidth=2,linestyle=line_styles[i])

# 添加标题和标签
plt.title('Performance Metrics for Different SEResNet Models')
plt.xlabel('Metrics')
plt.ylabel('Metric Values')
plt.legend()
plt.grid(True)
plt.savefig('different_SEResNet.png')
plt.show()
