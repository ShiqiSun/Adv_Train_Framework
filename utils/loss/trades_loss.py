import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim
from utils.register.registers import LOSSES
from utils.bound_temp import *

@LOSSES.register
def Trades(model, data, target, cfg, optimizer, *args):

    #load from cfg
    step_size = cfg.train.loss.step_size
    epsilon = cfg.train.loss.epsilon
    perturb_steps= cfg.train.loss.perturb_steps
    beta = cfg.train.loss.beta
    distance = cfg.train.loss.dis_type

     # define KL-loss
    criterion_kl = nn.KLDivLoss(size_average=False)
    model.eval()
    batch_size = len(data)
    # generate adversarial example
    x_adv = data.detach() + 0.001 * torch.randn(data.shape).to(cfg.local_rank).detach()
    if distance == 'l_inf':
        for _ in range(perturb_steps):
            x_adv.requires_grad_()
            with torch.enable_grad():
                loss_kl = criterion_kl(F.log_softmax(model(x_adv), dim=1),
                                       F.softmax(model(data), dim=1))
            grad = torch.autograd.grad(loss_kl, [x_adv])[0]
            x_adv = x_adv.detach() + step_size * torch.sign(grad.detach())
            x_adv = torch.min(torch.max(x_adv, data - epsilon), data + epsilon)
            x_adv = torch.clamp(x_adv, low_bound, up_bound)
    elif distance == 'l_2':
        delta = 0.001 * torch.randn(data.shape).cuda().detach()
        delta = Variable(delta.data, requires_grad=True)

        # Setup optimizers
        optimizer_delta = optim.SGD([delta], lr=epsilon / perturb_steps * 2)

        for _ in range(perturb_steps):
            adv = data + delta

            # optimize
            optimizer_delta.zero_grad()
            with torch.enable_grad():
                loss = (-1) * criterion_kl(F.log_softmax(model(adv), dim=1),
                                           F.softmax(model(data), dim=1))
            loss.backward()
            # renorming gradient
            grad_norms = delta.grad.view(batch_size, -1).norm(p=2, dim=1)
            delta.grad.div_(grad_norms.view(-1, 1, 1, 1))
            # avoid nan or inf if gradient is 0
            if (grad_norms == 0).any():
                delta.grad[grad_norms == 0] = torch.randn_like(delta.grad[grad_norms == 0])
            optimizer_delta.step()

            # projection
            delta.data.add_(data)
            delta.data.clamp_(0, 1).sub_(data)
            delta.data.renorm_(p=2, dim=0, maxnorm=epsilon)
        x_adv = Variable(data + delta, requires_grad=False)
    else:
        x_adv = torch.clamp(x_adv, low_bound, up_bound)
    model.train()


    x_adv = Variable(torch.clamp(x_adv, low_bound, up_bound), requires_grad=False)
    # zero gradient
    
    optimizer.zero_grad()
    # calculate robust loss
    logits = model(data) 
    loss_natural = F.cross_entropy(logits, target)
    loss_robust = (1.0 / batch_size) * criterion_kl(F.log_softmax(model(x_adv), dim=1),
                                                    F.softmax(model(data), dim=1))
    loss = loss_natural + beta * loss_robust
    return loss