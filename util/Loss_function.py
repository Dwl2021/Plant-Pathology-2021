import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1 - pt) ** self.gamma * BCE_loss
        return F_loss.mean()
    


class ArcFaceLoss(nn.Module):
    def __init__(self, scale=64.0, margin=0.50, easy_margin=False):
        super(ArcFaceLoss, self).__init__()
        self.scale = scale
        self.margin = margin
        self.easy_margin = easy_margin

        # 将 margin 转换为 Tensor
        margin_tensor = torch.tensor(margin)

        self.cos_m = torch.cos(margin_tensor)
        self.sin_m = torch.sin(margin_tensor)
        self.th = torch.cos(torch.pi - margin_tensor)
        self.mm = torch.sin(torch.pi - margin_tensor) * margin_tensor

    def forward(self, cosine, labels):

        # 对 cosine 进行 L2 归一化
        cosine = F.normalize(cosine, p=2, dim=1)

        sine = torch.sqrt(1.0 - torch.pow(cosine, 2) + 1e-7)
        phi = cosine * self.cos_m - sine * self.sin_m  # cos(theta + m)
        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where(cosine > self.th, phi, cosine - self.mm)

        one_hot = labels
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output *= self.scale
        loss = F.binary_cross_entropy_with_logits(output, labels.float())

        return loss


