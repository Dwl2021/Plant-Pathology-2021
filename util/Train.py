from tqdm import tqdm
import torch
from sklearn.metrics import precision_recall_fscore_support
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np


def Train(config,model, train_loader, val_loader, save_output=True):
    best_val_acc = 0.  # 记录最好的acc，用于保存模型
    val_predictions = []
    train_loss = []
    train_accuracy = []
    valid_loss = []
    valid_accuracy = []
    precision_list = []
    recall_list = []
    f1_list = []
    
    config.OPTIM = torch.optim.Adam(model.parameters(), lr=config.LR, weight_decay=config.WEIGHT_DECAY)
    
    for epoch in range(config.EPOCHS):
        model.train()
        ta, tl = train_fn(config, model, train_loader, epoch)
        model.eval()
        va, vl, val_preds, val_labels, precision, recall, f1 = valid_fn(config, model, val_loader, epoch)
        train_accuracy.append(ta)
        train_loss.append(tl)
        valid_loss.append(vl)
        valid_accuracy.append(va)
        precision_list.append(precision)
        recall_list.append(recall)
        f1_list.append(f1)
    
        print('Epoch {}, Train loss: {:.5f}, Train accuracy: {:.5f}, Val loss: {:.5f}, Val accuracy: {:.5f}'.format(epoch+1, tl, ta, vl, va))
        # print('Precision: {:.4f}, Recall: {:.4f}, F1 Score: {:.4f}'.format(precision, recall, f1))
        
        if va > best_val_acc:
            best_val_acc = va
            best_model_path = config.SAVE_DIR +config.MODEL_NAME +"_best.pt"
            torch.save(model.state_dict(), best_model_path)
    
    if save_output:
        # Figure 1 with two subplots
        plt.figure(figsize=(9, 4))
    
        # Plot loss
        plt.subplot(1, 2, 1)
        plt.plot(train_loss, label='Train Loss', color='#8ECFC9')
        plt.plot(valid_loss, label='Validation Loss', color='#FA7F6F')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.title('Training and Validation Loss')
    
        # Plot accuracy
        plt.subplot(1, 2, 2)
        plt.plot(train_accuracy, label='Train Accuracy', color='#FFBE7A')
        plt.plot(valid_accuracy, label='Validation Accuracy', color='#82B0D2')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.title('Training and Validation Accuracy')
    
        # Save the first figure with a different name
        plt.savefig(config.SAVE_DIR + config.MODEL_NAME + '_Loss_Accuracy.png')
    
        # Display the first figure
        plt.show()
    
        # Figure 2 with three subplots in a single row
        plt.figure(figsize=(9, 3))
    
        # Plot precision
        plt.subplot(1, 3, 1)
        plt.plot(precision_list, label='Precision', color='#BB9727')
        plt.xlabel('Epochs')
        plt.ylabel('Precision')
        plt.legend()
        plt.title('Precision')
    
        # Plot recall
        plt.subplot(1, 3, 2)
        plt.plot(recall_list, label='Recall', color='#32B897')
        plt.xlabel('Epochs')
        plt.ylabel('Recall')
        plt.legend()
        plt.title('Recall')
    
        # Plot F1 score
        plt.subplot(1, 3, 3)
        plt.plot(f1_list, label='F1 Score', color='#C76DA2')
        plt.xlabel('Epochs')
        plt.ylabel('F1 Score')
        plt.legend()
        plt.title('F1 Score')
    
        # Save the second figure with a different name
        plt.savefig(config.SAVE_DIR + config.MODEL_NAME + '_Precision_Recall_F1.png')
    return model, valid_loss[-1], valid_accuracy[-1], precision_list[-1], recall_list[-1], f1_list[-1]

def train_fn(config, model, train_loader, epoch):
    tr_loss = 0
    tr_accuracy = 0
    config.ACC_FUNC.reset()
    for images, labels in tqdm(train_loader, total=len(train_loader), desc="Epoch {} Training".format(epoch + 1)):
        images, labels = images.to(config.DEVICE), labels.to(config.DEVICE)
        images = images.unsqueeze(0)  ## just for navit
        config.OPTIM.zero_grad()
        pred = model(images)
        # calculate the loss
        loss = config.LOSS_FUNC(pred, labels.squeeze(-1))
        model.zero_grad()
        loss.backward()
        tr_loss += loss.item()
        config.ACC_FUNC.update(pred.cpu(), labels.cpu())
        config.OPTIM.step()
        
        # print("One image finished, running loss is" + str(tr_loss/TRAIN_SIZE))
    tr_accuracy = config.ACC_FUNC.compute().item()
    # print("One epoch finished,""Accuracy is",tr_accuracy,"Train loss is",tr_loss/TRAIN_SIZE)
    return tr_accuracy, tr_loss/config.TRAIN_SIZE



def valid_fn(config, model, val_loader, epoch):
    valid_loss = 0
    config.ACC_FUNC.reset()
    predictions = []
    true_labels = []

    with torch.no_grad():       
        for images, labels in tqdm(val_loader, total=len(val_loader), desc="Epoch {} Validing".format(epoch + 1)):
            images, labels = images.to(config.DEVICE), labels.to(config.DEVICE)
            pred = model(images)
            loss = config.LOSS_FUNC(pred, labels.squeeze(-1))
            valid_loss += loss.item()
            config.ACC_FUNC.update(pred.cpu(), labels.cpu())

            # Save predictions and true labels for later metric calculation
            predictions.append((torch.sigmoid(pred)>0.5).int().cpu().numpy().astype(np.int32))
            true_labels.append(labels.cpu().numpy().astype(np.int32))
            
    predictions = np.concatenate(predictions, axis=0)
    true_labels = np.concatenate(true_labels, axis=0)

    # Compute precision, recall, and F1 score
    precision, recall, f1, _ = precision_recall_fscore_support(true_labels, predictions, average='weighted')
    valid_accuracy = config.ACC_FUNC.compute().item()

    return valid_accuracy, valid_loss/config.VAL_SIZE, predictions, true_labels, precision, recall, f1
