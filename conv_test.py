from fairseq.models.wav2vec.wav2vec2 import Permute
from typing import Mapping
from omegaconf.omegaconf import open_dict
import torch
from fairseq.models.lightconv import LightConvEncoderLayer
from torch import nn
from omegaconf import DictConfig


def test():
    time, batch = 500 * 320, 16
    n_in, n_out, k, stride = 32, 512, 15, 5
    x = torch.randn(batch, n_in, time).cuda()
    print("Input", x.shape)

    conv = nn.Conv1d(n_in, n_out, k, stride=stride).cuda()
    y1 = conv(x)
    print("Conv", y1.shape)

    cfg = DictConfig({})
    with open_dict(cfg):
        cfg.encoder_embed_dim = n_in
        cfg.encoder_ffn_embed_dim = n_in * 4
        cfg.encoder_conv_dim = n_out
        cfg.encoder_glu = True
        cfg.encoder_conv_type = "lightweight"
        cfg.weight_softmax = True
        cfg.weight_dropout = 0.1
        cfg.relu_dropout = 0.1
        cfg.input_dropout = 0
        cfg.dropout = 0.1
        cfg.encoder_normalize_before = False
        cfg.encoder_attention_heads = 2
        
    padding_size = k // 2 if k % 2 == 1 else (k - 1) // 2
    lconv = nn.Sequential(
        Permute(2, 0, 1),
        LightConvEncoderLayer(cfg, kernel_size=k, padding_l=None),
        Permute(1, 2, 0),
        nn.AvgPool1d(kernel_size=stride),
        nn.Conv1d(n_in, n_out, kernel_size=1),
    ).cuda()
    y2 = lconv(x)
    print("LightConv", y2.shape)


if __name__ == "__main__":
    test()
