import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# 按照你的数据，创建一个包含指标的DataFrame
data = {
    'Model': ['ViT', 'DeepViT', 'CrossViT', 'Deit'],
    'Precision': [0.8881030597841272, 0.5958764231201099, 0.8860243757743124, 0.8715589522149357],
    'Recall': [0.838258164852255, 0.505443234836703, 0.8880248833592534, 0.8631415241057543],
    'F1': [0.861710185668124, 0.5234655394819211, 0.8855344880718915, 0.8639039234319914],
    'Test Accuracy': [0.824999988079071, 0.4650000035762787, 0.8033333420753479, 0.8166666626930237],
    'Valid Accuracy': [0.8333333134651184, 0.47333332896232605, 0.8500000238418579, 0.824999988079071],
}

df = pd.DataFrame(data)

# 剔除Model列，以Model为索引
df.set_index('Model', inplace=True)

# 创建热力图
plt.figure(figsize=(10,5 ))
sns.heatmap(df, annot=True, cmap="YlGnBu", fmt=".4f", linewidths=.5)
plt.title('Model Performance Heatmap')
plt.savefig('Model_Performance_Heatmap.png')
plt.show()
