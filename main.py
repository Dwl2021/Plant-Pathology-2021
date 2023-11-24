from pathlib import Path
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import sys
import os
from vit_pytorch.na_vit import NaViT

# custom module
from util.Data_loader import *
from util.Train import *
from util.Test import *
from util.Plant_Accuracy import *
from util.preprocessing import *

# models
from Models.ResNet import *
from Models.SEResnet import *
from Models.ViT import *
from torchvision.models import VisionTransformer
from super_gradients.training import models
from vit_pytorch.deepvit import DeepViT

args = sys.argv[1]

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

    EPOCHS = 100
    INPUT_HEIGHT = 224
    INPUT_WIDTH = 224

    # set the input heig/ht and width
    IMAGENET_MEAN = [0.485, 0.456, 0.406]
    IMAGENET_STD = [0.229, 0.224, 0.225]
    
    IMAGE_TYPE = '.jpg'
    BATCH_SIZE = 100
    MODEL_NAME = args

    LOSS_FUNC = nn.BCEWithLogitsLoss()
    ACC_FUNC = Plant_Accuracy()
    OPTIM = None
    LR = 0.0001
    WEIGHT_DECAY = 0
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    LABELS = ['complex', 'frog_eye_leaf_spot', 'healthy', 'powdery_mildew', 'rust', 'scab']
    NUM_CLASSES = len(LABELS)
    SAVE_DIR = "/root/CV_Model/result/"+f"{MODEL_NAME}/"
    LOG_TXT = "/root/CV_Model/log.txt"


def load_model(model_name):
    l_model = None
    if model_name == 'ResNet50':
        l_model = ResNet(model=50)
    elif model_name == 'ResNet101':
        l_model = ResNet(model=101)
    elif model_name == 'SEResNet34':
        l_model = SEResnet(model=34)
    elif model_name == 'SEResNet50':
        l_model = SEResnet(model=50)
    elif model_name == 'SEResNet101':
        l_model = SEResnet(model=101)
    elif model_name == 'SEResNet200':
        l_model = SEResnet(model=200)
    elif model_name == 'SG_ViT':
        l_model = models.get("vit_base", num_classes=6, pretrained_weights='imagenet')
    elif model_name == 'Torch_ViT':
        l_model = VisionTransformer(image_size=224, patch_size=16, num_layers=12, num_heads=12, hidden_dim=768, mlp_dim=3072, num_classes=6)
    elif model_name == "NaViT":
        l_model = NaViT(
                        image_size = 224,
                        patch_size = 16,
                        num_classes = 6,
                        dim = 1024,
                        depth = 6,
                        heads = 16,
                        mlp_dim = 2048,
                        dropout = 0.1,
                        emb_dropout = 0.1,
                        token_dropout_prob = 0.1  # token dropout of 10% (keep 90% of tokens)
                        )
    elif model_name == "DeepViT":
        DeepViT(
                image_size = 256,
                patch_size = 32,
                num_classes = 1000,
                dim = 1024,
                depth = 6,
                heads = 16,
                mlp_dim = 2048,
                dropout = 0.1,
                emb_dropout = 0.1
            )
    else:
        raise("Error!")
    return l_model

if __name__ == '__main__': 
    if not os.path.exists(config.SAVE_DIR):
        os.makedirs(config.SAVE_DIR)
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
    model = load_model(config.MODEL_NAME)
    model = model.to(config.DEVICE)

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
                   "\nDropout: " + str(0.1) +
                   "\nvalid loss: " + str(val_loss) +
                   "\nvalid accuracy: " + str(val_acc) +
                   "\nprecision: " + str(precision) +
                   "\nrecall: " + str(recall) +
                   "\nf1: " + str(f1) +
                   "\ntest accuracy: " + str(accuracy) +
                   "\ntest loss: " + str(loss) + "\n"+
                   "-------------------------------------\n"
                  )
    
    torch.cuda.empty_cache()
