import torch
import torch.nn as nn
from Models.extend.Crossvit import *

class Cross_ViT(nn.Module):
    def __init__(self, num_classes=6, pretrain_path=None):
        super(Cross_ViT, self).__init__()
        self.model = crossvit_small_224()
        if pretrain_path:
        # 使用 torch.load 加载模型权重
            state_dict = torch.load(pretrain_path)
            self.model.load_state_dict(state_dict)
        # 添加全连接层
        self.fc = nn.Sequential(
            nn.Linear(1000, 2048),
            nn.ReLU(),
            nn.Linear(2048, num_classes)
        )

    def forward(self, x):
        x = self.model(x)
        x = self.fc(x)
        return x
