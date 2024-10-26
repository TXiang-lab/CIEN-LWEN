from .dataset_catalog import DatasetCatalog
import torch
import torch.utils.data
from .plain_dataset import Dataset
from .collate_batch import my_collator

torch.multiprocessing.set_sharing_strategy('file_system')


def make_ddp_data_sampler(dataset, shuffle):
    sampler = torch.utils.data.distributed.DistributedSampler(dataset, shuffle=shuffle)
    return sampler


def make_ddp_train_loader(cfg):
    batch_size = cfg.train.batch_size
    shuffle = True
    drop_last = True
    dataset_name = cfg.train.dataset

    args = DatasetCatalog.get(dataset_name)  # SbdTrain
    dataset = Dataset(**args)

    sampler = make_ddp_data_sampler(dataset, shuffle)

    data_loader = torch.utils.data.DataLoader(
        dataset,
        sampler=sampler,
        batch_size=batch_size,
        num_workers=batch_size,
        pin_memory=False,
        drop_last=drop_last,
        collate_fn=my_collator
    )
    return data_loader


def make_data_sampler(dataset, shuffle):
    if shuffle:
        sampler = torch.utils.data.sampler.RandomSampler(dataset)
    else:
        sampler = torch.utils.data.sampler.SequentialSampler(dataset)
    return sampler


def make_batch_data_sampler(sampler, batch_size, drop_last):
    batch_sampler = torch.utils.data.sampler.BatchSampler(sampler, batch_size, drop_last)
    return batch_sampler


def make_test_loader(cfg, is_distributed=True):
    batch_size = 1
    shuffle = True if is_distributed else False
    drop_last = False
    dataset_name = cfg.val.dataset
    args = DatasetCatalog.get(dataset_name)  # SbdTrain
    dataset = Dataset(**args)

    sampler = make_data_sampler(dataset, shuffle)
    batch_sampler = make_batch_data_sampler(sampler, batch_size, drop_last)
    num_workers = 1
    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_sampler=batch_sampler,
        num_workers=num_workers,
        collate_fn=my_collator
    )
    return data_loader


def make_ddp_data_loader(is_train=True, is_distributed=False, cfg=None):
    if is_train:
        return make_ddp_train_loader(cfg), make_test_loader(cfg, is_distributed)
    else:
        return make_test_loader(cfg, is_distributed)