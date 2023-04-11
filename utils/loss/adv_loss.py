import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.register.registers import LOSSES
from utils.bound_temp import *

@LOSSES.register
def Adv(model, data, target, cfg, optimizer, *args):
    device = cfg.device
    epsilon = cfg.train.loss.epsilon
    perturb_steps = cfg.train.loss.perturb_steps
    step_size = cfg.train.loss.step_size

    x_adv = ce_adv(model, data,  target,
            perturb_steps=perturb_steps,
            epsilon=epsilon,
            step_size=step_size,)

    criterion_ce = F.cross_entropy 
    
    model.train()
    optimizer.zero_grad()
    loss = criterion_ce(model(data), target) + criterion_ce(model(x_adv), target)
    return loss

def ce_adv(model, 
            x, 
            target,
            perturb_steps=10,
            epsilon=0.3,
            step_size=0.01):
    
    criterion_ce = F.cross_entropy
    model.eval()
    x_adv = x.detach() + 0.001 * torch.randn(x.shape).cuda().detach()

    for step in range(perturb_steps):
        x_adv.requires_grad_()
        with torch.enable_grad():
                loss_ce = criterion_ce(model(x_adv), target)
        grad = torch.autograd.grad(loss_ce, [x_adv])[0]
        x_adv = x_adv.detach() + step_size * torch.sign(grad.detach())
        x_adv = torch.min(torch.max(x_adv, x - epsilon), x + epsilon)
        x_adv = torch.clamp(x_adv, low_bound, up_bound)
    
    return x_adv