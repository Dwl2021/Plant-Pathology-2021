import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import VisionTransformer

class Torch_ViT(nn.Module):
    def __init__(self, num_classes=6, pretrained=True):
        super(Torch_ViT, self).__init__()      
        self.model = VisionTransformer(
                                       image_size=224, 
                                       patch_size=16, 
                                       num_layers=12, 
                                       num_heads=12, 
                                       hidden_dim=768, 
                                       mlp_dim=3072, 
                                       num_classes=6
                                      )

    def forward(self, x):
        x = self.model(x)
        return x
