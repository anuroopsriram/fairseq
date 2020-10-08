from time import time

import torch

from fairseq.models.wav2vec import ConformerEncoderLayer, TransformerSentenceEncoderLayer


def params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def timeit(model, x):
    for _ in range(10):
        y = model(x)

    start = time()
    for _ in range(10):
        y = model(x)
    return time() - start


if __name__ == '__main__':
    trans = TransformerSentenceEncoderLayer().cuda()
    conf = ConformerEncoderLayer(embedding_dim=512).cuda()
    # confRelPos = ConformerEncoderLayer(
    #     embedding_dim=512, use_rel_posn_mha=True, num_relpos_embeds=16).cuda()

    B = 32
    T = 500
    C1 = 768
    C2 = 512
    x1 = torch.randn((T, B, C1)).cuda()
    x2 = torch.randn((T, B, C2)).cuda()

    print('Params Transformer', params(trans))
    print('Params Conformer', params(conf))
    # print('Params ConformerRelPos', params(confRelPos))

    print('Transformer', timeit(trans, x1))
    print('Conformer', timeit(conf, x2))
    # print('ConformerRelPos', timeit(confRelPos, x2))
