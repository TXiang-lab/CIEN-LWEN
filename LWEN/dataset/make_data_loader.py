from .dataset_catalog import DatasetCatalog
import torch
import torch.utils.data
from .plain_dataset import Dataset
from .collate_batch import my_collator
from . import samplers
torch.multiprocessing.set_sharing_strategy('file_system')



def make_data_sampler(dataset, shuffle):
    if shuffle:
        sampler = torch.utils.data.sampler.RandomSampler(dataset)
    else:
        sampler = torch.utils.data.sampler.SequentialSampler(dataset)
    return sampler


def make_batch_data_sampler(cfg, sampler, batch_size, drop_last, max_iter):
    batch_sampler = torch.utils.data.sampler.BatchSampler(sampler, batch_size, drop_last)
    if max_iter != -1:
        batch_sampler = samplers.IterationBasedBatchSampler(batch_sampler, max_iter)
    return batch_sampler

def make_data_loader(cfg, is_train=True, is_distributed=False, max_iter=-1):
    if is_train:
        batch_size = cfg.train.batch_size  # 80
        shuffle = True
        drop_last = True
    else:
        batch_size = cfg.val.batch_size
        shuffle = True if is_distributed else False
        drop_last = False
    dataset_name = cfg.train.dataset if is_train else cfg.val.dataset  # True

    args = DatasetCatalog.get(dataset_name)  # SbdTrain
    dataset = Dataset(**args)
    sampler = make_data_sampler(dataset, shuffle)
    batch_sampler = make_batch_data_sampler(cfg, sampler, batch_size, drop_last, max_iter)
    num_workers = cfg.train.num_workers  # 32
    collator = my_collator
    data_loader = torch.utils.data.DataLoader(dataset,
                                              batch_sampler=batch_sampler,
                                              num_workers=num_workers,
                                              collate_fn=collator)

    return data_loader
