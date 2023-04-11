import torch.optim as optim
from  utils.register.registers import OPTIMIZERS

@OPTIMIZERS.register
def SGD(cfg, model):
    optimizer = optim.SGD \
                (model.parameters(),
                  lr=cfg.lr,
                  momentum=cfg.momentum,
                  weight_decay=cfg.weight_decay)

    return optimizer

@OPTIMIZERS.register
def Adam(cfg, model):
    optimizer = optim.Adam \
                (model.parameters(),
                lr=cfg.lr,
                betas = (cfg.beta1, cfg.beta2),
                weight_decay=cfg.weight_decay
                )

    return optimizer