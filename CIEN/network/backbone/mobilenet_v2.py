import torch.nn as nn
from collections import OrderedDict


# 把channel变为8的整数倍
def _make_divisible(ch, divisor=8, min_ch=None):
    if min_ch is None:
        min_ch = divisor
    new_ch = max(min_ch, int(ch + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_ch < 0.9 * ch:
        new_ch += divisor
    return new_ch


# 定义基本的ConvBN+Relu
class baseConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, groups=1, stride=1):
        super(baseConv, self).__init__()
        pad = kernel_size // 2
        relu = nn.ReLU6(inplace=True)
        if kernel_size == 1 and in_channels > out_channels:
            relu = nn.Identity()
        self.baseConv = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride,
                      padding=pad, groups=groups, bias=False),
            nn.BatchNorm2d(out_channels),
            relu
        )

    def forward(self, x):
        out = self.baseConv(x)
        return out


# 定义残差结构
class residual(nn.Module):
    def __init__(self, in_channels, expand_rate, out_channels, stride):  # 输入和输出channel都要调整到8的整数倍
        super(residual, self).__init__()
        expand_channel = int(expand_rate * in_channels)  # 升维后的channel

        conv1 = baseConv(in_channels, expand_channel, 1)
        if expand_rate == 1:
            # 此时没有1*1卷积升维
            conv1 = nn.Identity()

        # channel1
        self.block1 = nn.Sequential(
            conv1,
            baseConv(expand_channel, expand_channel, 3, groups=expand_channel, stride=stride),
            baseConv(expand_channel, out_channels, 1)
        )

        if stride == 1 and in_channels == out_channels:
            self.has_res = True
        else:
            self.has_res = False

    def forward(self, x):
        if self.has_res:
            return self.block1(x) + x
        else:
            return self.block1(x)


# 定义mobilenetv2
class MobileNet_v2(nn.Module):
    def __init__(self, theta=1, num_classes=10, init_weight=True):
        super(MobileNet_v2, self).__init__()
        # [inchannel,t,out_channel,stride]
        # net_config = [[32, 1, 16, 1],
        #               [16, 6, 24, 2],
        #               [24, 6, 32, 2],
        #               [32, 6, 64, 2],
        #               [64, 6, 96, 1],
        #               [96, 6, 160, 2],
        #               [160, 6, 320, 1]]
        # self.repeat_num = [1, 2, 3, 4, 3, 3, 1]
        net_config = [[32, 1, 16, 1],
                      [16, 6, 24, 2],
                      [24, 6, 32, 2],
                      [32, 6, 64, 2],
                      [64, 6, 96, 1],
                      [96, 6, 160, 2]]
        self.repeat_num = [1, 2, 3, 4, 3, 3]
        module_dic = OrderedDict()
        setattr(self, 'first_conv', baseConv(3, _make_divisible(theta * 32), 3, stride=2))
        self.output_layer = []
        for idx, num in enumerate(self.repeat_num):
            stride = False
            parse = net_config[idx]
            if parse[-1] == 2:
                stride = True
            for i in range(num):
                setattr(self, 'bottleneck{}_{}'.format(idx, i + 1),
                        residual(_make_divisible(parse[0] * theta), parse[1], _make_divisible(parse[2] * theta),
                                 parse[3]))

                parse[0] = parse[2]
                parse[-1] = 1
            if stride:
                self.output_layer.append('bottleneck{}_{}'.format(idx, i + 1))

                # 初始化权重
        if init_weight:
            self.init_weight()

    def init_weight(self):
        for w in self.modules():
            if isinstance(w, nn.Conv2d):
                nn.init.kaiming_normal_(w.weight, mode='fan_out')
                if w.bias is not None:
                    nn.init.zeros_(w.bias)
            elif isinstance(w, nn.BatchNorm2d):
                nn.init.ones_(w.weight)
                nn.init.zeros_(w.bias)
            elif isinstance(w, nn.Linear):
                nn.init.normal_(w.weight, 0, 0.01)
                nn.init.zeros_(w.bias)

    def forward(self, x):
        y = []
        x = getattr(self, 'first_conv')(x)
        for idx, num in enumerate(self.repeat_num):
            for i in range(num):
                x = getattr(self, 'bottleneck{}_{}'.format(idx, i + 1))(x)
                if 'bottleneck{}_{}'.format(idx, i + 1) in self.output_layer:
                    y.append(x)
        return y
