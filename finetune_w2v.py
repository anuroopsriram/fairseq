from pathlib import Path

import numpy as np

import submit

wer_args = (
    '/datasets01_101/librispeech/021419/lm/4-gram.arpa',
    '/checkpoint/anuroops/data/libris/lab.960h/lexicon_ltr.lst',
    2, -1,
)


base_params = {
    'distributed-world-size': 24,
    'distributed-port': 13434,
    'save-dir': '/checkpoint/anuroops/fairseq/wav2vec/w2v.base.ft/',
    'fp16': True,
    'post-process': 'letter',
    'valid-subset': 'dev_other',
    'no-epoch-checkpoints': True,
    'best-checkpoint-metric': 'wer',
    'num-workers': 4,
    'max-update': 80000,
    'sentence-avg': True,
    'task': 'audio_pretraining',
    'arch': 'wav2vec_ctc',
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
}


def w2v_base_250k(args, params):
    args.name = args.name or 'w2v.base.250k.ft'
    args.nodes = 3
    return args, params


def w2v_base_400k(args, params):
    args.name = args.name or 'w2v.base.400k.ft'
    args.nodes = 3
    return args, params


def w2v_base_400k_lm(args, params):
    args.name = args.name or 'w2v.base.400k.ft.4glm'
    args.nodes = 3
    return args, params


def w2v_conformer_250k(args, params):
    args.name = args.name or 'w2v.conformer.250k.ft'
    args.nodes = 3
    return args, params


def w2v_conformer_relpos_250k(args, params):
    args.name = args.name or 'w2v.conformer.relpos.250k.ft'
    args.nodes = 3
    return args, params


def w2v_conformer_400k_lm(args, params):
    args.name = args.name or 'w2v.conformer.400k.ft.4glm'
    args.nodes = 3
    return args, params


#### Sweeps

@submit.register_sweep
def sweep_w2v_base_250k(base_args):
    lrs = [2e-05]
    checkpoint = Path('logs/w2v.base.250k/dim704.enclyrs17.lr0.0005')
    param_sweeps = [
        (
            f'ckpt{checkpoint.name}.lr{lr}',
            {
                'w2v-path': checkpoint / 'checkpoint_best.pt',
                'lr': lr,
            },
        )
        for lr in lrs
    ]
    submit.run_sweeps(w2v_base_250k, base_args, base_params, param_sweeps, dataset='lab.10h')


@submit.register_sweep
def sweep_w2v_base_400k(base_args):
    lrs = [2e-05]
    checkpoint = Path('logs/w2v.base.400k')
    param_sweeps = [
        (
            f'ckpt{checkpoint.name}.lr{lr}',
            {
                'w2v-path': checkpoint / 'checkpoint_best.pt',
                'lr': lr,
            },
        )
        for lr in lrs
    ]
    submit.run_sweeps(w2v_base_400k, base_args, base_params, param_sweeps, dataset='lab.10h')


@submit.register_sweep
def sweep_w2v_base_400k_4glm(base_args):
    lrs = [2e-05]
    checkpoint = Path('logs/w2v.base.400k')
    param_sweeps = [
        (
            f'ckpt{checkpoint.name}.lr{lr}',
            {
                'w2v-path': checkpoint / 'checkpoint_best.pt',
                'lr': lr,
                'wer-args': wer_args,
            },
        )
        for lr in lrs
    ]
    submit.run_sweeps(w2v_base_400k, base_args, base_params, param_sweeps, dataset='lab.10h')


@submit.register_sweep
def sweep_w2v_conformer_250k(base_args):
    lrs = [6e-05, 2e-05]
    checkpoints = [
        Path('logs/w2v.conformer.250k/dim576.enclyrs12.lr0.001'),
    ]
    param_sweeps = [
        (
            f'{checkpoint.name}/lr{lr}',
            {
                'w2v-path': checkpoint / 'checkpoint_best.pt',
                'lr': lr,
            },
        )
        for checkpoint in checkpoints
        for lr in lrs
    ]
    submit.run_sweeps(w2v_conformer_250k, base_args, base_params, param_sweeps, dataset='lab.10h')


@submit.register_sweep
def sweep_w2v_conformer_relpos_250k(base_args):
    lrs = [6e-05, 2e-05]
    checkpoints = [
        Path('logs/w2v.conformer.relpos.250k/dim512.enclyrs17.lr0.001.rpemb16'),
    ]
    param_sweeps = [
        (
            f'{checkpoint.name}/lr{lr}',
            {
                'w2v-path': checkpoint / 'checkpoint_best.pt',
                'lr': lr,
            },
        )
        for checkpoint in checkpoints
        for lr in lrs
    ]
    submit.run_sweeps(w2v_conformer_relpos_250k, base_args, base_params, param_sweeps, dataset='lab.10h')


@submit.register_sweep
def sweep_w2v_conformer_400k_4glm(base_args):
    lrs = [6e-05, 2e-05]
    checkpoints = [
        # Path('logs/w2v.conformer.400k/dim512.enclyrs17.lr0.001'),
        Path('logs/w2v.conformer.400k/dim512.enclyrs17.lr0.0005'),
    ]
    param_sweeps = [
        (
            f'{checkpoint.name}/lr{lr}',
            {
                'w2v-path': checkpoint / 'checkpoint_best.pt',
                'lr': lr,
                'wer-args': wer_args,
            },
        )
        for checkpoint in checkpoints
        for lr in lrs
    ]
    submit.run_sweeps(w2v_conformer_400k_lm, base_args, base_params, param_sweeps, dataset='lab.10h')


@submit.register_sweep
def sweep_w2v_conformer_400k_4glm_960h(base_args):
    lrs = [2e-05]
    checkpoints = [
        Path('logs/w2v.conformer.400k/dim512.enclyrs17.lr0.0005'),
    ]
    param_sweeps = [
        (
            f'{checkpoint.name}/lr{lr}',
            {
                'w2v-path': checkpoint / 'checkpoint_best.pt',
                'lr': lr,
                'wer-args': wer_args,
            },
        )
        for checkpoint in checkpoints
        for lr in lrs
    ]
    submit.run_sweeps(w2v_conformer_400k_lm, base_args, base_params, param_sweeps, dataset='lab.960h')


if __name__ == '__main__':
    parser = submit.create_parser()
    base_args = parser.parse_args()
    sweep_func = submit.get_sweep_func(base_args.sweep)
    sweep_func(base_args)
