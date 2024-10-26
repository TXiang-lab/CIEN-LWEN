from .base import *

model_dir = 'data/model/default_LWEN'
record_dir = 'data/record/default_LWEN'
result_dir = 'data/result/default_LWEN'
save_ep = 1
eval_ep = 1
# train
train.dataset = 'LISAP_train'
train.epoch = 400
train.batch_size = 1024
train.lr = 0.001
train.gamma = 0.1
train.warmup = True
train.warm_iter = 10
train.warm_factor = 0.1
train.milestones = [80, 160, 240, 320]
# eval_ep = 10
val.dataset = 'LISAP_val'
