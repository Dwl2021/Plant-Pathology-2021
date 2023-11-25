import torch
import torch.nn as nn
import torch.nn.functional as F
from super_gradients.training import models

class SG_ViT(nn.Module):
    def __init__(self, num_classes=6, pretrained=True):
        super(SG_ViT, self).__init__()      
        self.model = models.get("vit_base", num_classes=6, pretrained_weights='imagenet')

    def forward(self, x):
        x = self.model(x)
        return x
