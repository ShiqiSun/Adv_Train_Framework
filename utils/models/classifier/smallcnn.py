from collections import OrderedDict
import torch.nn as nn
from utils.register.registers import MODELS

@MODELS.register
class SmallCNN(nn.Module):
    def __init__(self, cfg, drop=0.5):
        super(SmallCNN, self).__init__()

        self._num_channels = cfg.num_channels != None and cfg.num_channels or 1
        self._num_classes = cfg.num_classes != None and cfg.num_classes or 10
        self._dropRate = cfg.dropRate != None and cfg.dropRate or drop

        activ = nn.ReLU(True)

        if self._num_channels == 3:
            self._out_channels = 5
        elif self._num_channels == 1:
            self._out_channels = 4

        self.feature_extractor = nn.Sequential(OrderedDict([
            ('conv1', nn.Conv2d(self._num_channels, 32, 3)),
            ('relu1', activ),
            ('conv2', nn.Conv2d(32, 32, 3)),
            ('relu2', activ),
            ('maxpool1', nn.MaxPool2d(2, 2)),
            ('conv3', nn.Conv2d(32, 64, 3)),
            ('relu3', activ),
            ('conv4', nn.Conv2d(64, 64, 3)),
            ('relu4', activ),
            ('maxpool2', nn.MaxPool2d(2, 2)),
        ]))

        self.classifier = nn.Sequential(OrderedDict([
            ('fc1', nn.Linear(64 * self._out_channels * self._out_channels, 200)),
            ('relu1', activ),
            ('drop', nn.Dropout(self._dropRate)),
            ('fc2', nn.Linear(200, 200)),
            ('relu2', activ),
            ('fc3', nn.Linear(200, self._num_classes)),
        ]))

        for m in self.modules():
            if isinstance(m, (nn.Conv2d)):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        nn.init.constant_(self.classifier.fc3.weight, 0)
        nn.init.constant_(self.classifier.fc3.bias, 0)

    def forward(self, input):
        features = self.feature_extractor(input)
        logits = self.classifier(features.view(-1, 64 * self._out_channels * self._out_channels))
        return logits