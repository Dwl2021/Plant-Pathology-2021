import torch
import torch.nn as nn

class Deit(nn.Module):
    def __init__(self, num_classes=6, pretrained=True):
        super(Deit, self).__init__()

        # Load the pre-trained DeiT model with the specified number of classes
        self.model = torch.hub.load('facebookresearch/deit', 'deit_base_patch16_224', pretrained=pretrained)
        self.fc =  nn.Sequential(
            nn.Linear(1000, 2048),
            nn.ReLU(),
            nn.Linear(2048, num_classes)
        )

    def forward(self, x):
        # Forward pass through the DeiT model
        x = self.model(x)
        x = self.fc(x)
        return x
