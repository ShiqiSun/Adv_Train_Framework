
import torch
from torch.autograd import Variable


def eval_atk_from_dataset(model, data_loader, device, attacker, cfg_atk, log=None):
    """
    eval attack effect from a dataloader 
    attacker: a function from utils.attack 
    cfg_attack: attack config from configs/Attack
    """
    model.eval()
    robust_err_total=0
    natural_err_total = 0

    sum = 0

    for data, target in data_loader:
        data, target = data.to(device), target.to(device)

        with torch.no_grad():
            out = model(data)
            err_natural = (out.data.max(1)[1] != target.data).float().sum()
        
        err_robust = attacker(model, data, target, device, cfg_atk)

        robust_err_total += err_robust
        natural_err_total += err_natural
        sum += data.shape[0]
        print(f"{err_robust}/{len(data)}:{(len(data)-err_robust)/len(data)},Sum:{sum}, Finished:{sum/len(data_loader):.2f}%, N{100*(1-natural_err_total/sum):.2f}%, R{100*(1-robust_err_total/sum):.2f}%" )
    log.logger.info(f'natural_right_total:{sum-natural_err_total} / total case:{sum} : {100*(1-natural_err_total/sum):.2f}%' )
    log.logger.info(f'robust_right_total:{sum-robust_err_total} / total case:{sum} : {100*(1-robust_err_total/sum):.2f}%', )
    pass