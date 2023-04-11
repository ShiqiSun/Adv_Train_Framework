import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from utils.register.registers import ATTACKS
from utils.bound_temp import *

@ATTACKS.register
def pgd_white(model,
                data,
                target,
                device,
                cfg_atk
                ):
    epsilon = cfg_atk.epsilon
    num_steps= cfg_atk.num_steps
    step_size= cfg_atk.step_size
    random = cfg_atk.random
    X_pgd = Variable(data.data, requires_grad=True)
    if random:
        random_noise = torch.FloatTensor(*X_pgd.shape).uniform_(-epsilon, epsilon).to(device)
        X_pgd = Variable(X_pgd.data + random_noise, requires_grad=True)

    for _ in range(num_steps):
        opt = optim.SGD([X_pgd], lr=1e-3) #1e-3
        opt.zero_grad()

        with torch.enable_grad():
            loss = F.cross_entropy(model(X_pgd), target)
        loss.backward()
        
        eta = step_size * X_pgd.grad.data.sign()
        X_pgd = Variable(X_pgd.data + eta, requires_grad=True)
        eta = torch.clamp(X_pgd.data - data.data, -epsilon, epsilon)
        X_pgd = Variable(data.data + eta, requires_grad=True)
        X_pgd = Variable(torch.clamp(X_pgd, low_bound, up_bound), requires_grad=True)
    # log = model(X_pgd)[:, 0]
    # maxi = log.max() 
    # idx_max = torch.argmax(log)
    # idx_min = torch.argmin(log)
    # mini = log.min()
    # dis = torch.norm(data[idx_max] - data[idx_min])
    # print(maxi, mini)
    # print(maxi - mini)
    # print((maxi - mini)/dis)
    # return maxi, mini
    err_pgd = (model(X_pgd).data.max(1)[1] != target.data).float().sum()
    return err_pgd

def pgd_whitbox_image(model,
                    data,
                    target,
                    device,
                    cfg_atk,
                ):
    data = data.to(device)
    target = target.to(device)
    epsilon = cfg_atk.epsilon
    num_steps= cfg_atk.num_steps
    step_size= cfg_atk.step_size
    random = cfg_atk.random
    X_pgd = Variable(data.data, requires_grad=True)
    model.eval()
    if random:
        random_noise = torch.FloatTensor(*X_pgd.shape).uniform_(-epsilon, epsilon).to(device)
        X_pgd = Variable(X_pgd.data + random_noise, requires_grad=True)

    for _ in range(num_steps):
        opt = optim.SGD([X_pgd], lr=1e-3)
        opt.zero_grad()

        with torch.enable_grad():
            loss = F.cross_entropy(model(X_pgd), target)
        loss.backward()
        # noise = torch.sign(torch.FloatTensor(*X_pgd.shape).uniform_(-1.0, 1.0).to(device))
        # # ones = torch.ones_like(noise)
        # # nones = -1*ones
        # # noise = torch.where(noise>0.0, ones, nones)
        # # print(noise, X_pgd.grad.data.sign())
        # eta_2 = step_size_2 * noise
        eta = step_size * X_pgd.grad.data.sign()
        # eta_2 = step_size_2*eta
        X_pgd = Variable(X_pgd.data + eta, requires_grad=True)
        # eta = torch.clamp(X_pgd.data - data.data, -epsilon, epsilon)
        # X_pgd = Variable(data.data + eta, requires_grad=True)
        # X_pgd = Variable(torch.clamp(X_pgd, -2.5, 2.5), requires_grad=True)
    # print(F.softmax(model(data))[0][label], F.softmax(model(X_pgd))[0][label])
    # print(model(data)[0][label], model(X_pgd)[0][label])
    # print("__________")
    loss = F.cross_entropy(model(X_pgd), target)

    return X_pgd, loss


def pgd_whitbox_image_3d(model,
                    data,
                    target,
                    device,
                    cfg_atk,
                    step_size_2,
                    noise
                ):
    data = data.to(device)
    target = target.to(device)
    epsilon = cfg_atk.epsilon
    num_steps= cfg_atk.num_steps
    step_size= cfg_atk.step_size
    random = cfg_atk.random
    X_pgd = Variable(data.data, requires_grad=True)

    # noise = torch.sign(torch.FloatTensor(*data.shape).uniform_(-1.0, 1.0).to(device))
    # ones = torch.ones_like(noise)
    # nones = -1*ones
    # noise = torch.where(noise>0.0, ones, nones)

    model.eval()
    # if random:
    #     random_noise = torch.FloatTensor(*X_pgd.shape).uniform_(-epsilon, epsilon).to(device)
    #     X_pgd = Variable(X_pgd.data + random_noise, requires_grad=True)

    for _ in range(num_steps):
        opt = optim.SGD([X_pgd], lr=1e-3)
        opt.zero_grad()

        with torch.enable_grad():
            loss = F.cross_entropy(model(X_pgd), target)
        loss.backward()
        eta_2 = step_size_2 * noise
        eta = step_size * X_pgd.grad.data.sign()
        X_pgd = Variable(X_pgd.data + eta + eta_2, requires_grad=True)
    loss = F.cross_entropy(model(X_pgd), target)

    return loss

@ATTACKS.register
def fgsm_attack(model,
                data,
                target,
                device,
                cfg_atk,
                ):
    
    cepsilon = cfg_atk.epsilon
    num_steps= cfg_atk.num_steps
    step_size= cfg_atk.step_size
    random = cfg_atk.random
    X_pgd = Variable(data.data, requires_grad=True)
    if random:
        random_noise = torch.FloatTensor(*X_pgd.shape).uniform_(-epsilon, epsilon).to(device)
        X_pgd = Variable(X_pgd.data + random_noise, requires_grad=True)

    for _ in range(num_steps):
        opt = optim.SGD([X_pgd], lr=1e-3) #1e-3
        opt.zero_grad()

        with torch.enable_grad():
            loss = F.cross_entropy(model(X_pgd), target)
        loss.backward()
        
        eta = step_size * X_pgd.grad.data.sign()
        X_pgd = Variable(X_pgd.data + eta, requires_grad=True)
        eta = torch.clamp(X_pgd.data - data.data, -epsilon, epsilon)
        X_pgd = Variable(data.data + eta, requires_grad=True)
        X_pgd = Variable(torch.clamp(X_pgd, 0.0, 1.0), requires_grad=True)
    # log = model(X_pgd)[:, 0]
    # maxi = log.max() 
    # idx_max = torch.argmax(log)
    # idx_min = torch.argmin(log)
    # mini = log.min()
    # dis = torch.norm(data[idx_max] - data[idx_min])
    # print(maxi, mini)
    # print(maxi - mini)
    # print((maxi - mini)/dis)
    # return maxi, mini
    err_pgd = (model(X_pgd).data.max(1)[1] != target.data).float().sum()
    return err_pgd