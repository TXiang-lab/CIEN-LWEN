import torch
from torch import nn
from .att import MultiHeadedAttention, PositionwiseFeedForward


class SelfAttention_block(torch.nn.Module):
    def __init__(self, c_in, num_heads):
        super(SelfAttention_block, self).__init__()
        self.att = MultiHeadedAttention(num_heads, c_in)
        self.feed_forward = PositionwiseFeedForward(c_in, c_in * 4)
        self.norm = torch.nn.LayerNorm(c_in, elementwise_affine=False)

    def forward(self, x):
        x = x + self.att(self.norm(x), self.norm(x), self.norm(x))
        return x + self.feed_forward(self.norm(x))


class CSNet(nn.Module):
    def __init__(self):
        super(CSNet, self).__init__()
        self.lin_proj = nn.Linear(4, 512)
        self.att = SelfAttention_block(512, 8)
        self.predict = nn.Sequential(
            nn.Linear(512, 128),
            nn.LeakyReLU(inplace=True),
            nn.Linear(128, 1)
        )
    def forward(self, x):
        x = self.lin_proj(x)
        x = self.att(x)

        x = x.mean(dim=1)
        return self.predict(x)
