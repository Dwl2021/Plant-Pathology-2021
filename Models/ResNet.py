import torch
import torch.nn as nn
import torch.nn.functional as F

class ResNet(nn.Module):
    def __init__(self, model=101,num_classes=6, pretrained=True):
        super(ResNet, self).__init__()
        
        # 使用torch.hub加载预训练的ResNet-50模型
        if model==101:
            self.model = torch.hub.load('pytorch/vision:v0.6.0', 'resnet101', pretrained=pretrained)
        elif model==50:
            self.model = torch.hub.load('pytorch/vision:v0.6.0', 'resnet50', pretrained=pretrained)
        else:
            raise ValueError("Unsupported model type. Supported types: 101, 50")

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
