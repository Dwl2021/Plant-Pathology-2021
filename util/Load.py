# models
from Models.ResNet import *
from Models.SEResnet import *
from Models.SG_ViT import *
from Models.Torch_ViT import *
from Models.DeepViT import *
from Models.CrossViT import *
from Models.Deit import *
from super_gradients.training import models
from vit_pytorch.deepvit import DeepViT

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
    elif model_name == 'SG_ViT':
        l_model = SG_ViT()
    elif model_name == 'Torch_ViT':
        l_model = Torch_ViT()
    elif model_name == "DeepViT":
        l_model = Deep_ViT()
    elif model_name == "CrossViT":
        l_model = Cross_ViT(pretrain_path='/root/crossvit_small_224.pth')
    elif model_name == "Deit":
        l_model = Deit()
    elif model_name == "ResNeXt":
        l_model = models.get("ResNext", num_classes=6, pretrained_weights='imagenet')
    else:
        raise("Models load Error!")
    return l_model


from util.Loss_function import *

def load_loss_function(loss_name):
    if loss_name=='BCEWithLogitsLoss':
        loss_function = nn.BCEWithLogitsLoss()
    elif loss_name=='FocalLoss':
        loss_function = FocalLoss()
    elif loss_name=='ArcFaceLoss':
        loss_function = ArcFaceLoss()
    else:
        raise("Loss function load Error!")
    return loss_function