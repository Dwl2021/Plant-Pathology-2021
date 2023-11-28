import matplotlib.pyplot as plt
import numpy as np

# Class names
categories = ['scab', 'healthy', 'frog_eye_leaf_spot', 'complex', 'rust', 'powdery_mildew']

# Training set label distribution data
label_counts = [645, 527, 490, 245, 234, 139]

# Model names
models = ['ResNet50', 'SEResNet50', 'ViT', 'DeiT']

# Accuracy data for each model
accuracies = [
    [98.5, 78.2, 90.4, 73.5, 94.2, 79.5],
    [97.8, 84.2, 90.4, 79.4, 92.3, 78.4],
    [97.0, 82.4, 93.7, 88.2, 92.3, 84.7],
    [96.5, 81.5, 95.2, 88.2, 92.3, 83.6]
]

marker_list = ['o', 's', '^', 'D']
line_styles = ['-', '-.', '--', ':']

# Create a subplo
fig, ax1 = plt.subplots(figsize=(6,5))

# Plot the bar chart first
ax1.bar(categories, label_counts, color='skyblue', alpha=0.5)
ax1.set_ylim(0, max(label_counts) + 50)  # Set the y-axis range starting from 0
ax1.set_ylabel('Sample Count')
#ax1.set_title('Training Set Label Distribution')

# Plot the line chart on top of the bar chart
ax2 = ax1.twinx()
for i in range(len(models)):
    ax2.plot(categories, accuracies[i], label=models[i], marker=marker_list[i], markersize=10, linewidth=2,
             linestyle=line_styles[i])

# Add legend and labels for the line chart
ax2.legend()
ax2.set_ylabel('Accuracy (%)')
#ax2.set_title('Accuracy of Each Model in Different Categories')
ax2.set_ylim(50, 110)  # Set the y-axis range starting from 50
ax2.grid(True)

# Adjust the layout
plt.tight_layout()

# Display the plot
plt.show()
