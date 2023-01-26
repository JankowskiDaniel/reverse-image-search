from torchvision import transforms

import torch
from torch import nn

from torchvision import models

class ImageEncoder(nn.Module):
    """
    Encode images to a fixed size vector
    """

    def __init__(self, model_name: str = "resnet50"):
        super(ImageEncoder, self).__init__()
        if model_name == "resnet18":
            resnet = models.resnet18(pretrained=True)
        elif model_name == "resnet50":
            resnet = models.resnet50(pretrained=True)
        else:
            raise ValueError("Incorrect type of ResNet architecture.")
        
        self.model = torch.nn.Sequential(*(list(resnet.children())[:-1]))

        for p in self.model.parameters():
            p.requires_grad = True

    def forward(self, images):
        return self.model(images).squeeze()
