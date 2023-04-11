from utils.register.register import Registry

MODELS = Registry("model")
LOSSES = Registry("loss")
DATASETS = Registry("dataset")
OPTIMIZERS = Registry("optimizer")
ATTACKS = Registry("attack")
HEADS = Registry("head")
BACKBONES = Registry("backbone")