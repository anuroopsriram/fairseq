import numpy as np

import submit

base_params = {
    'distributed-world-size': 24,
    'distributed-port': 13434,
    'save-dir': '/checkpoint/anuroops/fairseq/wav2vec/w2v.base.ft/',
    'fp16': True,
    # 'wer-args': ('/datasets01_101/librispeech/021419/lm/4-gram.arpa', '/checkpoint/anuroops/data/libris/lab/dict.ltr.txt', 2, -1),
    # 'post-process': 'letter',
    'valid-subset': 'dev_other',
    'no-epoch-checkpoints': True,
    'best-checkpoint-metric': 'wer',
    'num-workers': 4,
    'max-update': 80000,
    'sentence-avg': True,
    'task': 'audio_pretraining',
    'arch': 'conformer_seq2seq',
    'labels': 'ltr',
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
    'dropout': 0.0,
    'activation-dropout': 0.1,
    'criterion': 'cross_entropy',
    'attention-dropout': 0.0,
    'max-tokens': 1280000,
    'seed': 2337,
    'log-format': 'json',
    'log-interval': 500,
    'ddp-backend': 'no_c10d',
    'weight-decay': 1e-6,

    'lstm-hidden-size': 640,
    'num-lstm-layers': 1,
    'logmel': True,
    'in-d': 80,
    'specaug-prob': 0.8,
    'conv-feature-layers': [(512, 7, 2)] * 2,
}


def conformer_medium_ctc_wave(args, params):
    args.name = args.name or 'conformer.ctc.medium.wave'
    args.nodes = 4
    dim = 256
    params.update({
        'criterion': 'ctc',
        'arch': 'conformer_ctc',
        'conv-feature-layers': [(512, 10, 5)] + [(512, 3, 2)] * 4 + [(512, 2, 2)] * 2,
        'logmel': False,
        'in-d': 1,

        'encoder-layers': 16,
        'encoder-embed-dim': dim,
        'encoder-attention-heads': 4,

        'lr': 0.05 / np.sqrt(dim),
        'lr-scheduler': 'tri_stage',
        'warmup-steps': 10000,
        'hold-steps': 128000,
        'decay-steps': 160000,
    })
    del params['lstm-hidden-size']
    del params['num-lstm-layers']
    return args, params


def conformer_medium_ctc_logmel(args, params):
    args.name = args.name or 'conformer.ctc.medium.logmel'
    args.nodes = 4
    dim = 256
    params.update({
        # 'conv-feature-layers': [(512, 1, 1)] * 2,
        'conv-feature-layers': [(512, 7, 2)] * 1,
        'clip-norm': 10.,

        'criterion': 'ctc',
        'arch': 'conformer_ctc',

        'encoder-layers': 1,
        # 'encoder-layers': 16,
        'encoder-embed-dim': dim,
        'encoder-attention-heads': 4,

        'lr': 0.05 / np.sqrt(dim),
        'lr-scheduler': 'tri_stage',
        'warmup-steps': 10000,
        'hold-steps': 128000,
        'decay-steps': 160000,
    })
    del params['lstm-hidden-size']
    del params['num-lstm-layers']
    return args, params


def conformer_medium_seq2seq_logmel(args, params):
    args.name = args.name or 'conformer.s2s.medium.logmel'
    args.nodes = 4
    dim = 256
    params.update({
        'encoder-layers': 16,
        'encoder-embed-dim': dim,
        'encoder-attention-heads': 4,

        'lr': 0.05 / np.sqrt(dim),
        'lr-scheduler': 'tri_stage',
        'warmup-steps': 10000,
        'hold-steps': 128000,
        'decay-steps': 160000,
    })
    return args, params


#### Sweeps

@submit.register_sweep
def sweep_conformer_medium_ctc_wave(base_args):
    dim = 256
    lrs = [0.05]
    param_sweeps = [
        (
            f'medium.dim{dim}.lr{lr}',
            {
                'encoder-embed-dim': dim,
                'lr': lr / np.sqrt(dim),
            },
        )
        for lr in lrs
    ]
    submit.run_sweeps(conformer_medium_ctc_wave, base_args, base_params,
                      param_sweeps, dataset='lab.10h')


@submit.register_sweep
def sweep_conformer_medium_ctc_logmel(base_args):
    dim = 256
    lrs = [0.05]
    param_sweeps = [
        (
            f'medium.dim{dim}.lr{lr}',
            {
                'encoder-embed-dim': dim,
                'lr': lr / np.sqrt(dim),
            },
        )
        for lr in lrs
    ]
    submit.run_sweeps(conformer_medium_ctc_logmel, base_args, base_params,
                      param_sweeps, dataset='lab.10h')


@submit.register_sweep
def sweep_conformer_medium_seq2seq_logmel(base_args):
    dim = 256
    lrs = [0.05]
    param_sweeps = [
        (
            f'medium.dim{dim}.lr{lr}',
            {
                'encoder-embed-dim': dim,
                'lr': lr / np.sqrt(dim),
            },
        )
        for lr in lrs
    ]
    submit.run_sweeps(conformer_medium_seq2seq_logmel, base_args, base_params,
                      param_sweeps, dataset='lab.10h')


if __name__ == '__main__':
    parser = submit.create_parser()
    base_args = parser.parse_args()
    sweep_func = submit.get_sweep_func(base_args.sweep)
    sweep_func(base_args)

