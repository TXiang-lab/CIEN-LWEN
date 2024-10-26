from network.backbone.aggregation import DLAUp, IDAUp, fill_fc_weights, fill_up_weights
from network.backbone.mobilenet_v2 import MobileNet_v2
from torch import nn


class Mobile_seg(nn.Module):
    def __init__(self, heads, head_conv):
        super(Mobile_seg, self).__init__()
        self.base = MobileNet_v2()
        self.dla_up = DLAUp(0, [24, 32, 64, 160], [1, 2, 4, 8])
        self.ida_up = IDAUp(24, [24, 32, 64], [1, 2, 4])
        self.heads = heads
        for head in self.heads:
            classes = self.heads[head]
            if head_conv > 0:
                fc = nn.Sequential(
                    nn.Conv2d(24, head_conv,
                              kernel_size=3, padding=1, bias=True),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(head_conv, classes,
                              kernel_size=1, stride=1,
                              padding=1 // 2, bias=True))
                if 'hm' in head:
                    fc[-1].bias.data.fill_(-2.19)
                else:
                    fill_fc_weights(fc)
            else:
                fc = nn.Conv2d(24, classes,
                               kernel_size=1, stride=1,
                               padding=1 // 2, bias=True)
                if 'hm' in head:
                    fc.bias.data.fill_(-2.19)
                else:
                    fill_fc_weights(fc)
            self.__setattr__(head, fc)

    def forward(self, x):
        x = self.base(x)
        x = self.dla_up(x)

        y = []
        for i in range(3):
            y.append(x[i].clone())
        self.ida_up(y, 0, len(y))
        z = {}
        for head in self.heads:
            z[head] = self.__getattr__(head)(y[-1])
        return z, y[-1]
