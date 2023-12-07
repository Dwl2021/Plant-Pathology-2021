from pathlib import Path
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import sys
import os
from vit_pytorch.na_vit import NaViT

# util module
from util.Data_loader import *
from util.Train import *
from util.Test import *
from util.Plant_Accuracy import *
from util.preprocessing import *
from util.Load import *

import argparse

'''
for example:
python main.py --model_name CrossViT --epochs 100 --batch_size 200 --Loss_function ArcFaceLoss

'''

parser = argparse.ArgumentParser(description='Model training parameters')
parser.add_argument('--model_name', type=str, default=None, help='Name of the model')
parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
parser.add_argument('--batch_size', type=int, default=100, help='Batch size')
parser.add_argument('--Loss_function', type=str, default='ArcFaceLoss', help='Loss function')
args = parser.parse_args()

model_name = args.model_name
epochs = args.epochs
batch_size = args.batch_size
Loss_function = args.Loss_function


class config:
    TRAIN_INIT_DIR = Path('/root/plant_dataset/train/images')
    TEST_INIT_DIR = Path('/root/plant_dataset/test/images')
    VAL_INIT_DIR = Path('/root/plant_dataset/val/images')
    TRAIN_DIR = Path('/root/Plant-Pathology-2021/data/train/')
    TEST_DIR = Path('/root/Plant-Pathology-2021/data/test/')
    VAL_DIR = Path('/root/Plant-Pathology-2021/data/val/')

    CSV_TRAIN = Path('/root/Plant-Pathology-2021/data/train_label.csv')
    CSV_TEST = Path('/root/Plant-Pathology-2021/data/test_label.csv')
    CSV_VAL = Path('/root/Plant-Pathology-2021/data/val_label.csv')
    TRAIN_SIZE = 0
    TEST_SIZE = 0
    VAL_SIZE = 0

    EPOCHS = epochs
    INPUT_HEIGHT = 224
    INPUT_WIDTH = 224

    # set the input heig/ht and width
    IMAGENET_MEAN = [0.485, 0.456, 0.406]
    IMAGENET_STD = [0.229, 0.224, 0.225]
    
    IMAGE_TYPE = '.jpg'
    BATCH_SIZE = batch_size
    MODEL_NAME = model_name
    
    LOSS_FUNC = None
    LOSS_FUNC_NAME = Loss_function

    ACC_FUNC = Plant_Accuracy()
    OPTIM = None
    LR = 0.00001
    WEIGHT_DECAY = 0.000001
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    LABELS = ['complex', 'frog_eye_leaf_spot', 'healthy', 'powdery_mildew', 'rust', 'scab']
    NUM_CLASSES = len(LABELS)
    SAVE_DIR = "/root/Plant-Pathology-2021/result/"+f"{MODEL_NAME}_{LOSS_FUNC_NAME}/"
    LOG_TXT = "/root/Plant-Pathology-2021/result/log.txt"


if __name__ == '__main__': 
    if not os.path.exists(config.SAVE_DIR):
        os.makedirs(config.SAVE_DIR)
    # ---------------preprocess the saw image----------------
    # if you want to know more about the preprocessing, you 
    # can read the file at "/util/preprocessing.py"
    
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
    model = load_model(config.MODEL_NAME)
    model = model.to(config.DEVICE)
    config.LOSS_FUNC = load_loss_function(config.LOSS_FUNC_NAME)

    # ----------------------for test only--------------------
    '''
    print("Now begin testing...")
    pth_to_pt = "/root/best1.pt"
    model.load_state_dict(torch.load(pth_to_pt))
    model.eval()
    accuracy, loss= Test(config, model, test_loader, True)
    print("Test Accuracy: {:.4f}, Test Loss: {:.4f}".format(accuracy, loss))
    sys.exit()
    '''

    # ----------------------for train -------------------
    model,val_loss,val_acc,precision,recall,f1 = Train(config ,model, train_loader, val_loader, True)

     # ----------------------for test--------------------
    print("Now begin testing...")
    model.load_state_dict(torch.load( config.SAVE_DIR +config.MODEL_NAME +"_best.pt"))
    model.eval()
    accuracy, loss = Test(config, model, test_loader)
    print("Test Accuracy: {:.4f}, Test Loss: {:.4f}".format(accuracy, loss))
    
    with open(config.LOG_TXT, 'a') as file:
        file.write("Model: " + config.MODEL_NAME +
                   "\nEpochs: " + str(config.EPOCHS) +
                   "\nBase LR: " + str(config.LR) +
                   "\nWeight decay: " + str(config.WEIGHT_DECAY) +
                   "\nDropout: " + str(0.05) +
                   "\nvalid loss: " + str(val_loss) +
                   "\nvalid accuracy: " + str(val_acc) +
                   "\nprecision: " + str(precision) +
                   "\nrecall: " + str(recall) +
                   "\nf1: " + str(f1) +
                   "\ntest accuracy: " + str(accuracy) +
                   "\ntest loss: " + str(loss) + 
                   "\nloss function: " + config.LOSS_FUNC_NAME +"\n"+
                   "-------------------------------------\n"
                  )
    
    torch.cuda.empty_cache()
