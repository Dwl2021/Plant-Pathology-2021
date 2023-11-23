from pathlib import Path
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import sys

# custom module
from util.Data_loader import *
from util.Train import *
from util.Test import *
from util.Plant_Accuracy import *
from util.preprocessing import *

# mode
from ResNet import *
from SEResnet import *

class config:
    TRAIN_INIT_DIR = Path('/root/plant_dataset/train/images')
    TEST_INIT_DIR = Path('/root/plant_dataset/test/images')
    VAL_INIT_DIR = Path('/root/plant_dataset/val/images')
    TRAIN_DIR = Path('/root/CV_Model/data/train/')
    TEST_DIR = Path('/root/CV_Model/data/test/')
    VAL_DIR = Path('/root/CV_Model/data/val/')

    CSV_TRAIN = Path('/root/CV_Model/data/train_label.csv')
    CSV_TEST = Path('/root/CV_Model/data/test_label.csv')
    CSV_VAL = Path('/root/CV_Model/data/val_label.csv')
    TRAIN_SIZE = 0
    TEST_SIZE = 0
    VAL_SIZE = 0

    EPOCHES = 4
    INPUT_HEIGHT = 224
    INPUT_WIDTH = 224

    # set the input heig/ht and width
    IMAGENET_MEAN = [0.485, 0.456, 0.406]
    IMAGENET_STD = [0.229, 0.224, 0.225]
    
    IMAGE_TYPE = '.jpg'
    BATCH_SIZE = 50
    MODEL_NAME = 'ResNet101'

    LOSS_FUNC = nn.BCEWithLogitsLoss()
    ACC_FUNC = Plant_Accuracy()
    OPTIM = None
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    LABELS = ['complex', 'frog_eye_leaf_spot', 'healthy', 'powdery_mildew', 'rust', 'scab']
    NUM_CLASSES = len(LABELS)
    SAVE_DIR = "/root/CV_Model/"

if __name__ == '__main__': 
    # ---------------preprocess the saw image----------------
    
    # preprocessing(config)
    
    # ---------------create the dateloader--------------------
    ##  load the csv files
    train_df, valid_df, test_df, config = load_df(config);
    ## generate the transform
    train_transforms, val_transforms = get_transforms(config)
    ## load the datasets
    train_dataset = PlantDataset(config, train_df, transform = train_transforms)
    val_dataset = PlantDataset(config, valid_df, transform = val_transforms)
    test_dataset = PlantDataset(config, test_df, transform = val_transforms)
    ## transform to the dataloader
    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE)
    test_loader = DataLoader(test_dataset, batch_size=config.BATCH_SIZE)
    
    # ----------------------init the model--------------------
    
    ######### to change the model here ##########
    model = SEResnet().to(config.DEVICE)
    ############################################

    # ----------------------for test only--------------------
    print("Now begin testing...")
    pth_to_pt = "/root/SeResnet_best.pt"
    model.load_state_dict(torch.load(pth_to_pt))
    model.eval()
    accuracy, loss = test_fn(config, model, test_loader)
    print("Test Accuracy: {:.4f}, Test Loss: {:.4f}".format(accuracy, loss))
    sys.exit()

    
    # ----------------------for train and test------------------------
    best_val_acc = 0.#记录最好的acc，用于保存模型
    val_predictions = []
    train_loss = []
    train_accuracy = []
    valid_loss = []
    valid_accuracy = []
    config.OPTIM = optim.Adam(model.parameters(), lr=0.0001)
    for epoch in range(config.EPOCHES):
        model.train()
        ta, tl = train_fn(config,model, train_loader, epoch)
        model.eval()
        va, vl = valid_fn(config, model, val_loader, epoch)
        train_accuracy.append(ta)
        train_loss.append(tl)
        valid_loss.append(vl)
        valid_accuracy.append(va)
        # log_file.write(f"Epoch {epoch + 1}:\tTrain Acc {ta:.4f}\tTrain Loss {tl:.4f}\tVal Acc {va:.4f}\tVal Loss {vl:.4f}\n")
        print('Epoch {}, Train loss: {:.5f}, Train accuracy: {:.5f}, Val loss: {:.5f}, Val accuracy: {:.5f}'.format(epoch+1, tl, ta, vl, va))
        if va > best_val_acc:
            best_val_acc = va
            best_model_path = config.SAVE_DIR+"best.pt"
            torch.save(model.state_dict(), best_model_path)

    plt.figure(figsize=(12, 6))
    
    # Plot training loss with 'aquamarine' for train loss and 'dodgerblue' for validation loss
    plt.subplot(1, 2, 1)
    plt.plot(train_loss, label='Train Loss', color='aquamarine')
    plt.plot(valid_loss, label='Validation Loss', color='dodgerblue')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training and Validation Loss')
    
    # Plot training accuracy with 'salmon' for train accuracy and 'orange' for validation accuracy
    plt.subplot(1, 2, 2)
    plt.plot(train_accuracy, label='Train Accuracy', color='salmon')
    plt.plot(valid_accuracy, label='Validation Accuracy', color='orange')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title('Training and Validation Accuracy')
    plt.savefig(config.SAVE_DIR + config.MODEL_NAME+'.png')
    plt.show()
    
    
     # ----------------------for test--------------------
    print("Now begin testing...")
    model.load_state_dict(torch.load(config.SAVE_DIR+"best.pt"))
    model.eval()
    accuracy, loss = test_fn(config, model, test_loader)
    print("Test Accuracy: {:.4f}, Test Loss: {:.4f}".format(accuracy, loss))