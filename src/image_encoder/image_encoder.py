from torchvision import transforms

import torch
from torch import nn

from torchvision import models

class ResNetEncoder(nn.Module):
    def __init__(self, model: str = "resnet18") -> None:
        super(ResNetEncoder, self).__init__()
        if model == "resnet18":
            resnet = models.resnet18(pretrained=True)
        elif model == "resnet50":
            resnet = models.resnet50(pretrained=True)
        else:
            raise ValueError("Incorrect type of ResNet architecture.")
        self.model = torch.nn.Sequential(*(list(resnet.children())[:-1]))

    def forward(self, images):
        return self.model(images).squeeze()
