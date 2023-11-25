import timm
import torch.nn as nn
import torch

class Deit(nn.Module):
    def __init__(self, num_classes=6):
        super(Deit, self).__init__()

        # Load the pre-trained DeiT model
        self.base_model = timm.create_model('deit_small_patch16_224', pretrained=False)
        state_dict = torch.load("/root/CV_Model/deit_tiny_patch16_224-a1311bcf.pth")

        # Modify the output layer to match the specified number of classes
        self.base_model.head = nn.Linear(
            in_features=self.base_model.head.in_features,
            out_features=num_classes
        )

    def forward(self, x):
        # Forward pass through the DeiT model
        return self.base_model(x)

