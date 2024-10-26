# commen


gpus = [0, 1, 2, 3]
ct_score = 0.05
resume = True
save_ep = 5
eval_ep = 5
skip_eval = False
demo_path = ''
parallel = True
input_mode = 'norm_poly'
input_w = 768
input_h = 576


class train(object):
    dataset = 'LISAP_train'
    batch_size = 80
    epoch = 300
    gamma = 0.5
    lr = 0.0001
    milestones = [80, 120, 150, 170]
    num_workers = 64
    optim = 'adam'
    scheduler = ''
    warmup = False
    warm_iter = 5
    warm_factor = 0.3
    weight_decay = 0.0005


class val(object):
    dataset = 'LISAP_val'
    batch_size = 1
    epoch = -1


class test(object):
    dataset = 'LISAP_test'
    batch_size = 1
    epoch = -1
