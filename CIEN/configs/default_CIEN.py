from .base import commen, data, model, train, test

##p=1

model.heads['ct_hm'] = 1
train.batch_size = 64
train.epoch = 300
train.num_workers = 64
train.optimizer = {'name': 'adam', 'lr': 1e-4,
                   'weight_decay': 5e-4,
                   'milestones': [50, 100, 150, 200, 250, 270],
                   'gamma': 0.5}

commen.result_dir = 'data/result/default_CIEN'
commen.record_dir = 'data/record/default_CIEN'
commen.model_dir = 'data/model/default_CIEN'


class config(object):
    commen = commen
    data = data
    model = model
    train = train
    test = test
