import utils.loss.loss_registers
from utils.register.registers import LOSSES

def build_loss(cfg):

    loss = LOSSES[cfg.type]
    
    return loss