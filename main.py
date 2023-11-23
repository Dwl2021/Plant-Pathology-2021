from pathlib import Path
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import sys

# custom module
from util.Data_loader import *
from util.Train import *
from util.Test import *
from util.Plant_Accuracy import *
from util.preprocessing import *

# models
from Model.ResNet import *
from Model.SEResnet import *


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

    EPOCHES = 20
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
    '''
    print("Now begin testing...")
    pth_to_pt = "/root/SeResnet_best.pt"
    model.load_state_dict(torch.load(pth_to_pt))
    model.eval()
    accuracy, loss= Test(config, model, test_loader, True)
    print("Test Accuracy: {:.4f}, Test Loss: {:.4f}".format(accuracy, loss))
    sys.exit()
    '''
    

    # ----------------------for train -------------------


    model = Train(config ,model, train_loader, val_loader, True)

    
     # ----------------------for test--------------------
    print("Now begin testing...")
    model.load_state_dict(torch.load( config.SAVE_DIR +config.MODEL_NAME +"_best.pt"))
    model.eval()
    accuracy, loss = Test(config, model, test_loader)
    print("Test Accuracy: {:.4f}, Test Loss: {:.4f}".format(accuracy, loss))