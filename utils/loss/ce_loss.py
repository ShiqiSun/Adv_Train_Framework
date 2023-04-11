import torch.nn.functional as F
from utils.register.registers import LOSSES

@LOSSES.register
def CrossEntropy(model, data, target, *args):
    return F.cross_entropy(model(data), target)