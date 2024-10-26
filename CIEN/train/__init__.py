from .trainer import NetworkWrapper,Trainer
from .recorder import Recorder
from torch.optim.lr_scheduler import MultiStepLR
import torch
_optimizer_factory = {
    'adam': torch.optim.Adam,
    'sgd': torch.optim.SGD
}

def make_trainer(network, cfg):
    network = NetworkWrapper(network, with_dml=cfg.train.with_dml,
                          start_epoch=cfg.train.start_epoch, weight_dict=cfg.train.weight_dict)
    return Trainer(network)

def make_optimizer(net, cfg):
    cfg = cfg.train.optimizer
    params = []
    lr = cfg['lr']
    weight_decay = cfg['weight_decay']

    for key, value in net.named_parameters():
        if not value.requires_grad:
            continue
        params += [{"params": [value], "lr": lr, "weight_decay": weight_decay}]

    if 'adam' in cfg['name']:
        optimizer = _optimizer_factory[cfg['name']](params, lr, weight_decay=weight_decay)
    else:
        optimizer = _optimizer_factory[cfg['name']](params, lr, momentum=0.9)
    return optimizer

def make_lr_scheduler(optimizer, cfg):
    scheduler = MultiStepLR(optimizer, milestones=cfg.train.optimizer['milestones'],
                            gamma=cfg.train.optimizer['gamma'])
    return scheduler

def make_recorder(record_dir):
    return Recorder(record_dir=record_dir)