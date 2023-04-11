import torch
import torch.nn as nn
from utils.register.registers import MODELS
from collections import OrderedDict

@MODELS.register
class FFN(nn.Module):
    def __init__(self, cfg):
        super(FFN, self).__init__()
        dict_layers = OrderedDict([])
        self.__activ = nn.ReLU(True)

        dict_layers['conv1'] = nn.Linear(784, 1024)
        dict_layers['activ1'] =  self.__activ
        for i in range(cfg.layers - 1):
            dict_layers['conv'+str(i + 2)] = nn.Linear(1024, 1024)
            dict_layers['activ'+str(i + 2)] =  self.__activ
        dict_layers['out'] = nn.Linear(1024, 10)
        self.__classifier = nn.Sequential(dict_layers)
        

    def forward(self, x):
        x = torch.flatten(x, 1)
        x = self.__classifier(x)
        return x