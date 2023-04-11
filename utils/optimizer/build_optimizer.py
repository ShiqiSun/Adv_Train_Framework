
import utils.optimizer.optimizer_registers
from  utils.register.registers import OPTIMIZERS


def build_optimizer(cfg, model):
    optimizer = OPTIMIZERS[cfg.type](cfg, model)
    return optimizer
        

