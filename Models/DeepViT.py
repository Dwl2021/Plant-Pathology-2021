import torch
import torch.nn as nn
import torch.nn.functional as F
from vit_pytorch.deepvit import DeepViT

class Deep_ViT(nn.Module):
    def __init__(self, num_classes=6, pretrained=True):
        super(Deep_ViT, self).__init__()      
        self.model = DeepViT(image_size = 224,
                                patch_size = 16,
                                num_classes = 6,
                                dim = 1024,
                                depth = 6,
                                heads = 16,
                                mlp_dim = 2048,
                                dropout = 0.1,
                                emb_dropout = 0.1
                            )

    def forward(self, x):
        x = self.model(x)
        return x
