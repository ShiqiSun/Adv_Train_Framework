from audioop import bias
import math
from turtle import forward
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from utils.register.registers import MODELS

class BasicBlock(nn.Module):
    def __init__(self, in_planes, out_planes, stride, dropRate=0.0):
        super(BasicBlock, self).__init__()
        self._bn1 = nn.BatchNorm2d(in_planes)
        self._relu1 = nn.ReLU(inplace=True)
        self._conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self._bn2 = nn.BatchNorm2d(out_planes)
        self._relu2 = nn.ReLU(inplace=True)
        self._conv2 = nn.Conv2d(out_planes, out_planes, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self._droprate = dropRate
        self._equalInOut = (in_planes == out_planes)
        self._convShortcut = (not self._equalInOut) and nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride,
                                                                padding=0, bias=False) or None

    def forward(self, x):
        if not self._equalInOut:
            x = self._relu1(self._bn1(x))
        else:
            out = self._relu1(self._bn1(x))
        out = self._relu2(self._bn2(self._conv1(out if self._equalInOut else x)))
        if self._droprate > 0:
            out = F.dropout(out, p=self._droprate, training=self.training)
        out = self._conv2(out)
        return torch.add(x if self._equalInOut else self._convShortcut(x), out)

    
class NetworkBlock(nn.Module):
    def __init__(self, nb_layers, in_planes, out_planes, block, stride, dropRate=0.0) -> None:
        super().__init__()
        self._layer = self._make_layer(block, in_planes, out_planes, nb_layers, stride, dropRate)

    def _make_layer(self, block, in_planes, out_planes, nb_layers, stride, dropRate):
        layers = []
        for i in range(int(nb_layers)):
            layers.append(block(i==0 and in_planes or out_planes, out_planes, i == 0 and stride or 1, dropRate))
        return nn.Sequential(*layers)
    
    def forward(self, x):
        return self._layer(x)

@MODELS.register
class WideResNet(nn.Module):
    def __init__(self, cfg) -> None:
        super().__init__()

        self._depth = cfg.depth != None and cfg.depth or 34
        self._num_classes = cfg.num_classes != None and cfg.num_classes or 10
        self._widen_factor = cfg.widen_factor != None and cfg.widen_factor or 10
        self._dropRate = cfg.dropRate != None and cfg.dropRate or 0.0
        self._stride_list = cfg.stride_list != None and cfg.stride_list or [1, 1, 2, 2] #for CIFAR10
        assert ((self._depth - 4)%6 == 0)
        # n = (depth - 4)/6
        
        self.feature_extractor = feature_extractor(self._depth, self._widen_factor, \
                                                        self._dropRate, self._stride_list)
        self.classifier = classifier(self._depth, self._num_classes, self._widen_factor, self._dropRate)
        
        self.para_init()

    def para_init(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()
    
    def forward(self, x):
        out = self.feature_extractor(x)
        out = self.classifier(out)
        return out


class feature_extractor(nn.Module):
    def __init__(self, depth=28, widen_factor=10, dropRate=0.0, stride_list=[1, 1 ,2 ,2]) -> None:
        super().__init__()
        
        _nChannels = [16, 16*widen_factor, 32*widen_factor, 64*widen_factor]
        n = (depth - 4)/6
        block = BasicBlock

        self._conv1 = nn.Conv2d(3, _nChannels[0], kernel_size=3, stride=stride_list[0],
                                padding=1, bias=False)
        
        self._block1 = NetworkBlock(n, _nChannels[0], _nChannels[1], block, stride_list[1], dropRate)
        self._block2 = NetworkBlock(n, _nChannels[1], _nChannels[2], block, stride_list[2], dropRate)
        self._block3 = NetworkBlock(n, _nChannels[2], _nChannels[3], block, stride_list[3], dropRate)

    def forward(self, x):
        out = self._conv1(x)
        out = self._block1(out)
        out = self._block2(out)
        out = self._block3(out)
        return out


class classifier(nn.Module):
    def __init__(self, depth=28, num_classes=10, widen_factor=10, dropRate=0.0) -> None:
        super().__init__()

        _nChannels = [16, 16*widen_factor, 32*widen_factor, 64*widen_factor]
        self._bn1 = nn.BatchNorm2d(_nChannels[3], momentum=0.9)
        self._relu = nn.ReLU(inplace=True)
        self.fc = nn.Linear(_nChannels[3], num_classes)
        self.nChannels = _nChannels[3]
    
    def forward(self, x):
        out = self._relu(self._bn1(x))
        out = F.avg_pool2d(out, 8)
        out = out.view(-1, self.nChannels)
        out = self.fc(out)
        return out