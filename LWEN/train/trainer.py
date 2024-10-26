import torch.nn as nn
import torch
import time
import datetime
import tqdm
from torch.nn import DataParallel


class NetworkWrapper(nn.Module):
    def __init__(self,cfg, net):
        super(NetworkWrapper, self).__init__()

        self.net = net

        self.w_crit = torch.nn.SmoothL1Loss()
        self.mode=cfg.input_mode
    def forward(self, batch):
        predict_weights = self.net(batch[self.mode])

        scalar_stats = {}
        loss = 0

        w_loss = self.w_crit(batch['weight'], predict_weights)
        scalar_stats.update({'w_loss': w_loss})
        loss += w_loss

        image_stats = {}
        output = {'predict_weights': predict_weights, 'true_weights': batch['weight'], 'catids': batch['cls_id']}
        return output, loss, scalar_stats, image_stats


class Trainer(object):
    def __init__(self, network):
        network = network.cuda()
        self.network = network

    def reduce_loss_stats(self, loss_stats):
        reduced_losses = {k: torch.mean(v) for k, v in loss_stats.items()}
        return reduced_losses

    def train(self, epoch, data_loader, optimizer, recorder):
        max_iter = len(data_loader)
        self.network.train()
        end = time.time()
        train_loss_stats = {}
        for iteration, batch in enumerate(data_loader):
            if batch is None:
                continue
            data_time = time.time() - end
            iteration = iteration + 1
            recorder.step += 1

            # batch = self.to_cuda(batch)
            output, loss, loss_stats, image_stats = self.network(batch)
            # training stage: loss; optimizer; scheduler
            loss = loss.mean()
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_value_(self.network.parameters(), 40)  # 梯度裁剪 [-40,40]
            optimizer.step()

            # data recording stage: loss_stats, time, image_stats
            loss_stats = self.reduce_loss_stats(loss_stats)
            recorder.update_loss_stats(loss_stats)
            # 记录整个epoch损失
            for k, v in loss_stats.items():
                train_loss_stats.setdefault(k, 0)
                train_loss_stats[k] += v

            batch_time = time.time() - end
            end = time.time()
            recorder.batch_time.update(batch_time)
            recorder.data_time.update(data_time)

            if iteration % 20 == 0 or iteration == (max_iter - 1):
                # print training state
                eta_seconds = recorder.batch_time.global_avg * (max_iter - iteration)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                lr = optimizer.param_groups[0]['lr']
                memory = torch.cuda.max_memory_allocated() / 1024.0 / 1024.0

                training_state = '  '.join(['eta: {}', '{}', 'lr: {:.6f}', 'max_mem: {:.0f}'])
                training_state = training_state.format(eta_string, str(recorder), lr, memory)
                print(training_state)

                # record loss_stats and image_dict
                recorder.update_image_stats(image_stats)
                recorder.record('train')

        return train_loss_stats['w_loss'] / max_iter

    def val(self, epoch, data_loader, evaluator=None, recorder=None):
        self.network.eval()
        torch.cuda.empty_cache()
        val_loss_stats = {}
        data_size = len(data_loader)
        total_time = 0
        for batch in tqdm.tqdm(data_loader):

            if batch is None:
                continue

            for k in batch:
                batch[k] = batch[k].cuda()

            with torch.no_grad():
                start = time.time()
                output, loss, loss_stats, image_stats = self.network(batch)
                total_time += time.time() - start
                if evaluator is not None:
                    evaluator.evaluate(output, batch)

            loss_stats = self.reduce_loss_stats(loss_stats)
            for k, v in loss_stats.items():
                val_loss_stats.setdefault(k, 0)
                val_loss_stats[k] += v
        print(total_time / len(data_loader), '{} FPS'.format(len(data_loader) / total_time))
        loss_state = []
        for k in val_loss_stats.keys():
            val_loss_stats[k] /= data_size
            loss_state.append('{}: {:.4f}'.format(k, val_loss_stats[k]))
        print(loss_state)

        if evaluator is not None:
            result, fig_stat = evaluator.summarize()
            val_loss_stats.update(result)

        if recorder:
            recorder.record('val', epoch, val_loss_stats, image_stats)
            recorder.record_fig('val', epoch, fig_stat)