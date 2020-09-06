import numpy as np
from copy import deepcopy
from pathlib import Path

import submit


base_params = {
    'distributed-world-size': 24,
    'distributed-port':13434,
    'save-dir': '/checkpoint/anuroops/fairseq/wav2vec/w2v.small.ft.logmel/',
    'fp16': True,
    # 'wer-args': ('/datasets01_101/librispeech/021419/lm/4-gram.arpa', '/checkpoint/anuroops/data/libris/lab/dict.ltr.txt', 2, -1),
    'post-process': 'letter',
    'valid-subset': 'dev_other',
    'no-epoch-checkpoints': True,
    'best-checkpoint-metric': 'wer',
    'num-workers': 4,
    'max-update': 80000,
    'sentence-avg': True,
    'task': 'audio_pretraining',
    'arch': 'wav2vec_ctc',

    'no-pretrained-weights': True,
    'w2v-path': 'logs/w2v.small.logmel.ks1.logmelfix/lr0.0005.clipnorm0/checkpoint_best.pt',

    'labels': 'ltr',
    'apply-mask': False,
    'mask-selection': 'static',
    'mask-other': 0,
    'mask-length': 10,
    'mask-prob': 0.5,
    'layerdrop': 0.,
    'mask-channel-selection': 'static',
    'mask-channel-other': 0,
    'mask-channel-length': 64,
    'mask-channel-prob': 0.5,
    'zero-infinity': True,
    'feature-grad-mult': 0.0,
    'freeze-finetune-updates': 10000,
    'validate-after-updates': 10000,
    'optimizer': 'adam',
    'adam-betas': (0.9, 0.98),
    'adam-eps': 1e-09,
    'lr': 2e-03,
    'lr-scheduler': 'tri_stage',
    'warmup-steps': 10000,
    'hold-steps': 128000,
    'decay-steps': 160000,
    'final-lr-scale': 0.05,
    'final-dropout': 0.0,
    'dropout': 0.1,
    'activation-dropout': 0.,
    'criterion': 'ctc',
    'attention-dropout': 0.0,
    'max-tokens': 1280000,
    'seed': 2337,
    'log-format': 'json',
    'log-interval': 500,
    'ddp-backend': 'no_c10d',
    'logmel': True,
    # 'weight-decay': 1e-6,
    'specaug-prob': 0.4,
}


if __name__ == '__main__':
    parser = submit.create_parser()
    base_args = parser.parse_args()
    if base_args.nodes != 4:
        base_params['update-freq'] = 32 / base_args.nodes / 8

    dims = [
        # (16, 144),
        (16, 256),
        (17, 512)
    ]
    # lrs = [0.05, 0.005]
    lrs = [0.05]
    # lrs = [0.002, 0.0005]
    ckptroot = 'logs/w2v.small.conformer.logmel.dummy'
    param_sweeps = [
        (
            f'dim{dim}.lyr{lyr}.lr{lr}',
            {
                'w2v-path': f'{ckptroot}/dim{dim}.lyr{lyr}.lr0.0005/checkpoint_best.pt',
                'lr': lr / np.sqrt(dim),
            },
        )
        for lyr, dim in dims
        for lr in lrs
    ]
    for name, overrides in param_sweeps:
        assert Path(overrides['w2v-path']).exists(), overrides['w2v-path']
        args = deepcopy(base_args)
        args.name = f'{base_args.name}/{name}'
        params = deepcopy(base_params)
        params.update(**overrides)
        print(args.name, overrides)
        submit.main(args, params, 'lab.960h')

