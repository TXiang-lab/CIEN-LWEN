from .trainer import NetworkWrapper,Trainer
import torch
from .recorder import Recorder
from torch.optim.lr_scheduler import MultiStepLR
from .scheduler import WarmupMultiStepLR

def make_trainer(cfg,network):
    network = NetworkWrapper(cfg,network)
    return Trainer(network)

_optimizer_factory = {
    'adam': torch.optim.Adam,
    'sgd': torch.optim.SGD
}

def make_optimizer(cfg, net):
    params = []
    lr = cfg.train.lr
    weight_decay = cfg.train.weight_decay

    for key, value in net.named_parameters():
        if not value.requires_grad:
            continue
        params += [{"params": [value], "lr": lr, "weight_decay": weight_decay}]

    if 'adam' in cfg.train.optim:
        optimizer = _optimizer_factory[cfg.train.optim](params, lr, weight_decay=weight_decay)
    else:
        optimizer = _optimizer_factory[cfg.train.optim](params, lr, momentum=0.9)

    return optimizer

def make_recorder(cfg):
    return Recorder(cfg)

def make_lr_scheduler(cfg, optimizer):
    if cfg.train.warmup:
        scheduler = WarmupMultiStepLR(optimizer, cfg.train.milestones, cfg.train.gamma, cfg.train.warm_factor, cfg.train.warm_iter, 'linear')
    else:
        scheduler = MultiStepLR(optimizer, milestones=cfg.train.milestones, gamma=cfg.train.gamma)
    return scheduler
