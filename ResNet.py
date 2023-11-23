import torch
import torch.nn as nn
import torch.nn.functional as F

class ResNet101(nn.Module):
    def __init__(self, num_classes=6, pretrained=True):
        super(ResNet101, self).__init__()
        
        # 使用torch.hub加载预训练的ResNet-50模型
        self.model = torch.hub.load('pytorch/vision:v0.6.0', 'resnet101', pretrained=pretrained)
        # 修改全连接层，将输出特征数修改为num_classes
        # self.model.fc = nn.Linear(in_features=2048, out_features=num_classes, bias=True)
        in_features = self.model.fc.in_features
        self.model.fc = nn.Sequential(
            nn.Linear(in_features, 4096),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(4096, 2048),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(2048, num_classes)
        )

    def forward(self, x):
        x = self.model(x)
        return x
