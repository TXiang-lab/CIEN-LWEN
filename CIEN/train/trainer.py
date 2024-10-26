import time
import datetime
import tqdm
import torch.nn as nn
from .utils import FocalLoss, sigmoid, OTLoss
import torch


def collect_training(poly, ct_01):
    batch_size = ct_01.size(0)
    poly = torch.cat([poly[i][ct_01[i]] for i in range(batch_size)], dim=0)
    return poly


class NetworkWrapper(nn.Module):
    def __init__(self, net, with_dml=True, start_epoch=10, weight_dict=None):
        super(NetworkWrapper, self).__init__()
        self.with_dml = with_dml
        self.net = net
        self.ct_crit = FocalLoss()
        self.py_crit = OTLoss()
        self.weight_dict = weight_dict
        self.start_epoch = start_epoch


    def forward(self, batch):
        output = self.net(batch['inp'], batch)
        if 'test' in batch['meta']:
            return output

        scalar_stats = {}
        loss = 0.
        ct_01 = batch['ct_01'].byte()
        img_gt_polys = collect_training(batch['img_gt_polys'], ct_01)
        output.update({'img_gt_polys': img_gt_polys * 4})

        ct_loss = self.ct_crit(sigmoid(output['ct_hm']), batch['ct_hm'])
        scalar_stats.update({'ct_loss': ct_loss})
        loss += ct_loss

        num_polys = len(output['poly_init'])
        if num_polys == 0:
            init_py_loss = torch.sum(output['poly_init']) * 0.
            coarse_py_loss = torch.sum(output['poly_coarse']) * 0.
            final_py_loss = torch.sum(output['poly_final']) * 0.
        else:
            init_py_loss = self.py_crit(output['poly_init'], output['img_gt_polys'])
            coarse_py_loss = self.py_crit(output['poly_coarse'], output['img_gt_polys'])
            final_py_loss = self.py_crit(output['poly_final'], output['img_gt_polys'])
        scalar_stats.update({'init_py_loss': init_py_loss})
        scalar_stats.update({'coarse_py_loss': coarse_py_loss})
        scalar_stats.update({'final_py_loss': final_py_loss})
        loss += init_py_loss * self.weight_dict['init']
        loss += coarse_py_loss * self.weight_dict['coarse']
        loss += final_py_loss * self.weight_dict['final']
        scalar_stats.update({'loss': loss})

        return output, loss, scalar_stats


class Trainer(object):
    def __init__(self, network):
        network = network.cuda()
        self.network = network

    def reduce_loss_stats(self, loss_stats):
        reduced_losses = {k: torch.mean(v) for k, v in loss_stats.items()}
        return reduced_losses

    def to_cuda(self, batch):
        for k in batch:
            if k == 'meta':
                continue
            if isinstance(batch[k], tuple):
                batch[k] = [b.cuda() for b in batch[k]]
            else:
                batch[k] = batch[k].cuda()
        return batch

    def train(self, epoch, data_loader, optimizer, recorder):
        max_iter = len(data_loader)
        self.network.train()
        end = time.time()
        for iteration, batch in enumerate(data_loader):
            data_time = time.time() - end
            iteration = iteration + 1
            recorder.step += 1

            batch = self.to_cuda(batch)
            batch.update({'epoch': epoch})
            output, loss, loss_stats = self.network(batch)
            
            loss = loss.mean()
            optimizer.zero_grad()
            
            loss.backward()
            torch.nn.utils.clip_grad_value_(self.network.parameters(), 40)
            optimizer.step()

            loss_stats = self.reduce_loss_stats(loss_stats)
            recorder.update_loss_stats(loss_stats)

            batch_time = time.time() - end
            end = time.time()
            recorder.batch_time.update(batch_time)
            recorder.data_time.update(data_time)

            if iteration % 20 == 0 or iteration == (max_iter - 1):
                eta_seconds = recorder.batch_time.global_avg * (max_iter - iteration)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                lr = optimizer.param_groups[0]['lr']
                memory = torch.cuda.max_memory_allocated() / 1024.0 / 1024.0

                training_state = '  '.join(['eta: {}', '{}', 'lr: {:.6f}', 'max_mem: {:.0f}'])
                training_state = training_state.format(eta_string, str(recorder), lr, memory)
                print(training_state)

                recorder.record('train')

    def val(self, epoch, data_loader, evaluator=None, recorder=None):
        self.network.eval()
        torch.cuda.empty_cache()
        val_loss_stats = {}
        for batch in tqdm.tqdm(data_loader):
            for k in batch:
                if k != 'meta':
                    batch[k] = batch[k].cuda()

            batch.update({'epoch': epoch})
            with torch.no_grad():
                output = self.network(batch)
                if evaluator is not None:
                    evaluator.evaluate(output, batch)

        if evaluator is not None:
            result = evaluator.summarize()
            val_loss_stats.update(result)

        if recorder:
            recorder.record('val', epoch, val_loss_stats)

