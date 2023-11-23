from tqdm import tqdm
import torch
import numpy as np
from sklearn.metrics import multilabel_confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import sys

def Test(config, model, test_loader, save_confusion=True):
    test_loss = 0
    predictions = []
    true_labels = []
    pred_saw = []
    config.ACC_FUNC.reset()

    with torch.no_grad():
        for images, labels in tqdm(test_loader, total=len(test_loader), desc="Testing "):
            images, labels = images.to(config.DEVICE), labels.to(config.DEVICE)
            pred = model(images)
            loss = config.LOSS_FUNC(pred, labels.squeeze(-1))
            test_loss += loss.item()

            config.ACC_FUNC.update(pred.cpu(), labels.cpu())

            # Save predictions and true labels as lists
            pred_saw.append(torch.sigmoid(pred).cpu().numpy().astype(np.int32))
            predictions.append((torch.sigmoid(pred)>0.5).int().cpu().numpy().astype(np.int32))
            true_labels.append(labels.cpu().numpy().astype(np.int32))

    # Convert lists to NumPy arrays
    predictions = np.concatenate(predictions, axis=0)
    true_labels = np.concatenate(true_labels, axis=0)
    pred_saw = np.concatenate(pred_saw, axis=0)
    
    test_accuracy = config.ACC_FUNC.compute().item()

    if save_confusion:
        plot_confusion_matrix(true_labels, predictions, save_path=config.SAVE_DIR + "test_confusion.png")
        # plot_kdeplot(pred_saw, save_path=config.SAVE_DIR + "test_kdeplot.png")

    return test_accuracy, test_loss / config.TEST_SIZE

def plot_kdeplot(predictions,save_path):
    print(predictions)
    plt.figure(figsize=(10, 6))
    sns.kdeplot(data=predictions, fill=True)
    plt.xlabel('Prediction Probability')
    plt.ylabel('Density')
    plt.title('Predicted Label Probability Distribution')
    plt.savefig(save_path)

def plot_confusion_matrix(true_labels, predictions, save_path):
    cm = multilabel_confusion_matrix(true_labels, predictions)
    num_classes = cm.shape[0]

    rows = 2
    cols = 3

    plt.figure(figsize=(11, 7))
    for i in range(num_classes):
        plt.subplot(rows, cols, i + 1)
        sns.heatmap(cm[i], annot=True, fmt="d", cmap="Blues", xticklabels=["0", "1"], yticklabels=["0", "1"])
        plt.title(f"Class {i + 1}")

    plt.suptitle("Multilabel Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")

    plt.tight_layout()  # Adjust layout for better spacing
    plt.savefig(save_path)


