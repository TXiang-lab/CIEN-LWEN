import torch
import torch.nn as nn
import copy
import torch.nn.functional as F
import math


def clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


def attention(query, key, value, dropout=None):
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    p_attn = scores.softmax(dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn


class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.):
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value):
        nbatches = query.size(0)
        query, key, value = [
            lin(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
            for lin, x in zip(self.linears, (query, key, value))
        ]

        x, self.attn = attention(
            query, key, value, dropout=self.dropout
        )

        x = (
            x.transpose(1, 2)
            .contiguous()
            .view(nbatches, -1, self.h * self.d_k)
        )
        return self.linears[-1](x)


class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))

class ContourSelfAttention(torch.nn.Module):
    def __init__(self, c_in, num_heads):
        super(ContourSelfAttention, self).__init__()
        self.att = MultiHeadedAttention(num_heads, c_in)
        self.feed_forward = PositionwiseFeedForward(c_in, c_in * 4)

        self.lin = torch.nn.Linear(in_features=c_in,
                                   out_features=2, bias=True)
        self.norm1 = torch.nn.LayerNorm(c_in)
        self.norm2 = torch.nn.LayerNorm(c_in)
        self.norm3 = torch.nn.LayerNorm(c_in)

    def forward(self, x):
        x = x + self.att(self.norm1(x), self.norm1(x), self.norm1(x))
        x = x + self.feed_forward(self.norm2(x))
        return self.lin(self.norm3(x))
