from time import time

import torch

from fairseq.models.wav2vec import ConformerEncoderLayer


def params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def timeit(model, x):
    for _ in range(10):
        y = model(x)

    start = time()
    for _ in range(10):
        y = model(x)
    return time() - start


def main(x, **kwargs):
    print('Params', kwargs)
    conf = ConformerEncoderLayer(use_rel_posn_mha=True, **kwargs).cuda()
    print('Params ConformerRelPos', params(conf))
    time = timeit(conf, x)
    print(time, "\n")


if __name__ == '__main__':
    B = 32
    T = 500
    C = 768
    x = torch.randn((T, B, C)).cuda()

    # for ks in [4, 6, 12, 24, 32]:
    for ks in [3, 5, 7, 11, 15, 32]:
        main(x, kern_size=ks)
