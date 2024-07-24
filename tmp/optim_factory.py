import torch.nn as nn
from torch.optim import AdamW, Adam, SGD


def param_groups_weight_decay(
        model: nn.Module,
        weight_decay=1e-5,
        no_weight_decay_list=()
):
    no_weight_decay_list = set(no_weight_decay_list)
    decay = []
    no_decay = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        if param.ndim <= 1 or name.endswith(".bias") or name in no_weight_decay_list:
            no_decay.append(param)
        else:
            decay.append(param)

    return [
        {'params': no_decay, 'weight_decay': 0.},
        {'params': decay, 'weight_decay': weight_decay}]


def create_optimizer(
        model: nn.Module,
        lr: float,
        opt: str,
        weight_decay: float,
        no_weight_decay_list = ()
):
    param_list = param_groups_weight_decay(model, weight_decay, no_weight_decay_list)
    if opt == 'SGD':
        optimizer = SGD(param_list, lr=lr, momentum=0.9)
    elif opt == 'AdamW':
        optimizer = AdamW(param_list, lr=lr, betas=(0.9, 0.999))
    elif opt == 'Adam':
        optimizer = Adam(param_list, lr=lr)
    else:
        optimizer = AdamW(param_list, lr=lr)

    return optimizer




