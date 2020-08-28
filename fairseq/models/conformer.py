from typing import Any

import torch

from fairseq.modules import MultiheadAttention

from fairseq.models import BaseFairseqModel
from torch import nn


class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)


class ConvolutionalModule(nn.Module):
    def __init__(self, channels, kern_size, dropout):
        super().__init__()
        pad = (kern_size - 1) // 2
        self.net = nn.Sequential(
            nn.LayerNorm(channels),
            nn.Conv1d(channels, channels, kernel_size=1),
            nn.GLU(),
            nn.Conv1d(channels, channels, kern_size, padding=pad, groups=channels),
            nn.BatchNorm1d(channels),
            Swish(),
            nn.Conv1d(channels, channels, kernel_size=1),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return x + self.net(x.transpose(1, 2)).transpose(1, 2)


class FeedForwardModule(nn.Module):
    def __init__(self, dim, dropout, residual_scale=0.5):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Conv1d(dim, dim, kernel_size=1),
            Swish(),
            nn.Dropout(dropout),
            nn.Conv1d(dim, dim, kernel_size=1),
            nn.Dropout(dropout),
        )
        self.residual_scale = residual_scale

    def forward(self, x):
        return x + self.residual_scale * self.net(x)


class SelfAttentionModule(nn.Module):
    def __init__(self):
        super().__init__()
        # self.net = nn.Sequential(
        #     nn.LayerNorm()
        # )

    def forward(self, x):
        return x


class RelativePositionMultiHeadAttention(nn.Module):
    def forward(self, x):
        pass


class ConformerModule(nn.Module):
    def __init__(self, channels, dropout, kern_size):
        super().__init__()
        self.feedforward1 = FeedForwardModule(channels, dropout)
        self.selfattention = RelativePositionMultiHeadAttention()
        self.conv = ConvolutionalModule(channels, kern_size, dropout)
        self.feedforward2 = FeedForwardModule(channels, dropout)
        self.norm = nn.LayerNorm(channels)

    def forward(self, x):
        x = self.feedforward1(x)
        x = self.selfattention(x)
        x = self.conv(x)
        x = self.feedforward2(x)
        return self.norm(x)


class ConformerModel(BaseFairseqModel):

    @staticmethod
    def add_args(parser):
        parser.add_argument('--in-d', type=int, default=1)
        parser.add_argument('--channels', type=int, default=256)
        parser.add_argument('--kern-size', type=int, default=3)
        parser.add_argument('--dropout', type=float, default=0.1)
        parser.add_argument('--num-conformers', type=int, default=16)

    def __init__(self, args):
        super().__init__()
        self.net = nn.Sequential(
            # Subsampling Conv
            nn.Conv1d(args.in_d, args.channels, args.kern_size, stride=2),
            nn.GLU(),
            nn.LayerNorm(args.channels),
            nn.Conv1d(args.channels, args.channels, args.kern_size, stride=2),
            nn.GLU(),
            nn.LayerNorm(args.channels),
            # Linear
            nn.Conv1d(args.channels, args.channels, kernel_size=1),
            nn.Dropout(args.dropout),
            # Conformer blocks
            *(ConformerModule(args.channels, args.dropout, args.kern_size)
              for _ in range(args.num_conformers))
        )

    @classmethod
    def build_model(cls, args, task):
        pass

    def forward(self, x):
        return self.net(x)
