from turtle import forward
import torchvision.models as models
import torch.nn as nn
from utils.register.registers import MODELS

class v_model(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self._model = None

    def forward(self, x):
        assert self._model != None
        x = self._model(x)
        return x

@MODELS.register
class wideresnet_v(v_model):
    def __init__(self, cfg) -> None:
        super().__init__()
        self._model = models.wide_resnet50_2(pretrained=cfg.pretrained)

@MODELS.register
class vit_b_16_v(v_model):
    def __init__(self, cfg) -> None:
        super().__init__()
        self._model = models.vit_b_16(pretrained=cfg.pretrained)
