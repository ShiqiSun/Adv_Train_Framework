from turtle import forward
import torch.nn as nn
from torchvision import models
from utils.register.registers import MODELS

@MODELS.register
class ResNet50(nn.Module):

    def __init__(self, cfg) -> None:
        super().__init__()
        if "pretrained" not in cfg:
            cfg.pretrained = False
        self.model = models.resnet50(pretrained=cfg.pretrained)

    def forward(self, x):
        x = self.model(x)
        return x