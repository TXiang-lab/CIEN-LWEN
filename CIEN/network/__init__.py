import torch.nn as nn
import torch
from .backbone.mobile_seg import Mobile_seg
from .evolution.evo import Decode

class Network(nn.Module):
    def __init__(self, cfg=None):
        super(Network, self).__init__()

        self.test_stage = cfg.test.test_stage
        self.backbone = Mobile_seg(cfg.model.heads, cfg.model.head_conv)
        self.train_decoder = Decode(num_point=cfg.commen.points_per_poly, init_stride=cfg.model.init_stride,
                                    coarse_stride=cfg.model.coarse_stride, down_sample=cfg.commen.down_ratio,
                                    min_ct_score=cfg.test.ct_score)

    def forward(self, x, batch=None):
        output, cnn_feature = self.backbone(x)
        if 'test' not in batch['meta']:
            self.train_decoder(batch, cnn_feature, output, is_training=True)
        else:
            with torch.no_grad():
                if self.test_stage == 'init':
                    ignore = True
                else:
                    ignore = False
                self.train_decoder(batch, cnn_feature, output, is_training=False, ignore=ignore)
        return output
