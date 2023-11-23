import torch
import torch.nn as nn
import torch.nn.functional as F


# 搭建基于SENet的Conv Block和Identity Block的网络结构
class Block(nn.Module):
    def __init__(self, in_channels, filters, stride = 1, is_1x1conv = False):
        super(Block, self).__init__()
 
        filter1, filter2, filter3 = filters
        self.is_1x1conv = is_1x1conv # 判断是否是Conv Block
        self.relu = nn.ReLU(inplace = True) # RELU操作
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, filter1, kernel_size = 1, stride = stride, bias = False),
            nn.BatchNorm2d(filter1),
            nn.ReLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(filter1, filter2, kernel_size=3, stride = 1, padding = 1, bias=False),
            nn.BatchNorm2d(filter2),
            nn.ReLU()
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(filter2, filter3, kernel_size = 1, stride = 1, bias=False),
            nn.BatchNorm2d(filter3),
        )
        if is_1x1conv:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, filter3, kernel_size = 1, stride = stride, bias = False),
                nn.BatchNorm2d(filter3)
            )
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)), # 全局平均池化
            nn.Conv2d(filter3, filter3 // 16, kernel_size=1), # 16表示r，filter3//16表示C/r，这里用卷积层代替全连接层
            nn.ReLU(),
            nn.Conv2d(filter3 // 16, filter3, kernel_size=1),
            nn.Sigmoid()
        )
 
    def forward(self, x):
        x_shortcut = x
        x1 = self.conv1(x) # 执行第一Block操作
        x1 = self.conv2(x1) # 执行中间Block操作
        x1 = self.conv3(x1) # 执行最后Block操作
        x2 = self.se(x1)  # 利用SENet计算出每个通道的权重大小
        x1 = x1 * x2  # 对原通道进行加权操作
        if self.is_1x1conv:  # Conv Block进行额外的卷积归一化操作
            x_shortcut = self.shortcut(x_shortcut)
 
        x1 = x1 + x_shortcut  # Add操作
        x1 = self.relu(x1)  # ReLU操作
        return x1
 
class SEResnet(nn.Module):
    def __init__(self, num_classes=6, model=101):
        super(SEResnet, self).__init__()
        classes = num_classes
        if model == 101:
            num = [3, 4, 23, 3]
        elif model == 50:
            num = [3, 4, 6, 3]
        elif model == 18:
            num = [2, 2, 2, 2]
        elif model == 34:
            num = [3, 4, 6, 3]
        elif model == 200:
            # Custom configuration with more blocks in the final stage
            num = [3, 24, 36, 3]
        else:
            raise ValueError("Unsupported ResNet model. Supported models are 18, 34, 50, 101, and 200.")

        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size = 7, stride = 2, padding = 3, bias = False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 3, stride = 2, padding = 1)
        )
 
        # Stage1
        filters = (64, 64, 256)
        self.Stage1 = self._make_layer(in_channels = 64, filters = filters, num = num[0], stride = 1)
 
        # Stage2
        filters = (128, 128, 512) 
        self.Stage2 = self._make_layer(in_channels = 256, filters = filters, num = num[1], stride = 2)
 
        # Stage3
        filters = (256, 256, 1024)
        self.Stage3 = self._make_layer(in_channels = 512, filters = filters, num = num[2], stride = 2)
 
        # Stage4
        filters = (512, 512, 2048)
        self.Stage4 = self._make_layer(in_channels = 1024, filters = filters, num = num[3], stride = 2)

        self.global_average_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Sequential(
            nn.Linear(2048, classes)
        )
 
 
    # 形成单个Stage的网络结构
    def _make_layer(self, in_channels, filters, num, stride = 1):
        layers = []
        block_1 = Block(in_channels, filters, stride = stride, is_1x1conv = True)
        layers.append(block_1)
        for i in range(1, num):
            layers.append(Block(filters[2], filters, stride = 1, is_1x1conv = False))
        return nn.Sequential(*layers)
 
 
    def forward(self, x):
 
        # Stem Block环节
        x = self.conv1(x)
 
        # 执行四个Stage环节
        x = self.Stage1(x)
        x = self.Stage2(x)
        x = self.Stage3(x)
        x = self.Stage4(x)
 
        # 执行Subsequent Processing环节
        x = self.global_average_pool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
 
        return x
 