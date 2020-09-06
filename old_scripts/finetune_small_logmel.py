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
    'w2v-path': '/checkpoint/anuroops/fairseq/wav2vec/w2v.small.logmel/lr3e-06/checkpoint_best.pt',
    'labels': 'ltr',
    'apply-mask': True,
    'mask-selection': 'static',
    'mask-other': 0,
    'mask-length': 10,
    'mask-prob': 0.5,
    'layerdrop': 0.1,
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
    'adam-eps': 1e-08,
    'lr': 2e-05,
    'lr-scheduler': 'tri_stage',
    'warmup-steps': 8000,
    'hold-steps': 32000,
    'decay-steps': 40000,
    'final-lr-scale': 0.05,
    'final-dropout': 0.0,
    'dropout': 0.0,
    'activation-dropout': 0.1,
    'criterion': 'ctc',
    'attention-dropout': 0.0,
    'max-tokens': 1280000,
    'seed': 2337,
    'log-format': 'json',
    'log-interval': 500,
    'ddp-backend': 'no_c10d',
    'logmel': True,
    # 'in-d': 80,
    # 'clip-norm': 0.0001,
}


if __name__ == '__main__':
    parser = submit.create_parser()
    base_args = parser.parse_args()
    if base_args.nodes != 3:
        base_params['update-freq'] = 24 / base_args.nodes / 8

    paths = ['lr0.0005.clipnorm0']
    # paths = ["lr0.00075.clipnorm0", "lr0.001.clipnorm0"]
    # lrs = [6e-6, 2e-5, 6e-5, 2e-4]
    # lrs = [6e-6]
    lrs = [6e-5, 2e-6, 6e-6]
    clipnorms = [0]
    # dropouts = [0.1, 0.2]
    # layerdrops = [0.1, 0.2]
    dropouts = [0.05]
    layerdrops = [0.1]
    attentiondropouts = [0, 0.05]
    sweeps = [
        (lr, clipnorm, do, ld, ad)
        for lr in lrs
        for clipnorm in clipnorms
        for do in dropouts
        for ld in layerdrops
        for ad in attentiondropouts
    ]
    for path in paths:
        for lr, clipnorm, do, ld, ad in sweeps:
            params = deepcopy(base_params)
            checkpoint = Path(f'/checkpoint/anuroops/fairseq/wav2vec/w2v.small.logmel.ks1.logmelfix/')
            checkpoint = checkpoint / path / 'checkpoint_best.pt'
            assert Path(checkpoint).exists()
            params['w2v-path'] = str(checkpoint)

            params['lr'] = lr
            params['clip-norm'] = clipnorm
            params['dropout'] = do
            params['layerdrop'] = ld
            params['attention-dropout'] = ad

            args = deepcopy(base_args)
            args.name = f'{base_args.name}/{checkpoint.parent.name}/lr{lr}.clipnorm{clipnorm}.do{do}.ld{ld}'

            print(checkpoint, args.name, lr, clipnorm)
            submit.main(args, params, 'lab.10h')
