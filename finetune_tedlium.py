from pathlib import Path

import numpy as np

import submit
import copy

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
    'valid-subset': 'dev',
    'no-epoch-checkpoints': True,
    'best-checkpoint-metric': 'wer',
    'num-workers': 4,
    'max-update': 320000,
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
    # 'freeze-finetune-updates': 10000,
    'freeze-finetune-updates': 0,
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


def w2v_base_250k_ted(args, params):
    args.name = args.name or 'w2v.base.250k.robust.ft'
    args.data = '/checkpoint/anuroops/data/tedlium/'
    args.nodes = 3
    return args, params


def w2v_base_250k_libris(args, params):
    args.name = args.name or 'w2v.base.250k.robust.ft'
    args.data = '/checkpoint/anuroops/data/libris/'
    args.nodes = 3
    return args, params


def w2v_base_250k_ted_libris(args, params):
    args.name = args.name or 'w2v.base.4x250k.robust.ft'
    args.data = '/checkpoint/anuroops/data/ted.libris/'
    args.nodes = 3
    return args, params


#### Sweeps

@submit.register_sweep
def sweep_w2v_base_250k_scratch(base_args):
    lrs = [5e-04, 2e-04]
    checkpoints = [
        Path('logs/w2v.base.4x250k.ted/lr0.0003.ted.unlab/')
    ]
    param_sweeps = [
        (
            f'ckpt{checkpoint.name}.lr{lr}',
            {
                'no-pretrained-weights': True,
                'w2v-path': checkpoint / 'checkpoint_best.pt',
                'lr': lr,
            },
        )
        for checkpoint in checkpoints
        for lr in lrs
    ]
    base_args.nodes = 4
    base_args.name = "w2v.base.250k.robust.scratch"
    submit.run_sweeps(w2v_base_250k_ted, base_args, base_params, param_sweeps, dataset='ted.lab')


@submit.register_sweep
def sweep_w2v_base_250k(base_args):
    # lrs = [6e-05, 2e-05]
    lrs = [1e-04]
    checkpoints = [
        Path('logs/w2v.base.4x250k.ted/lr0.0003.ted.unlab/')
    ]
    param_sweeps = [
        (
            f'ckpt{checkpoint.name}.lr{lr}',
            {
                'w2v-path': checkpoint / 'checkpoint_best.pt',
                'lr': lr,
            },
        )
        for checkpoint in checkpoints
        for lr in lrs
    ]
    submit.run_sweeps(w2v_base_250k_ted, base_args, base_params, param_sweeps, dataset='ted.lab')

    # base_params2 = copy.deepcopy(base_params)
    # base_params2["valid-subset"] = "dev_other"
    # submit.run_sweeps(w2v_base_250k_libris, base_args, base_params2, param_sweeps, dataset='lab.960h')


@submit.register_sweep
def sweep_w2v_base_250k2(base_args):
    lrs = [1e-04]
    checkpoints = [
        Path('logs/w2v.base.4x250k.ted.libris/lr0.0003.unlab/')
    ]
    param_sweeps = [
        (
            f'ckpt{checkpoint.name}.lr{lr}',
            {
                'w2v-path': checkpoint / 'checkpoint_best.pt',
                'lr': lr,
            },
        )
        for checkpoint in checkpoints
        for lr in lrs
    ]
    # submit.run_sweeps(w2v_base_250k_ted, base_args, base_params, param_sweeps, dataset='ted.lab')
    submit.run_sweeps(w2v_base_250k_ted_libris, base_args, base_params, param_sweeps, dataset='ted.libris.lab.full')

    base_params2 = copy.deepcopy(base_params)
    base_params2["valid-subset"] = "dev_other"
    submit.run_sweeps(w2v_base_250k_libris, base_args, base_params2, param_sweeps, dataset='lab.960h')


if __name__ == '__main__':
    parser = submit.create_parser()
    base_args = parser.parse_args()
    sweep_func = submit.get_sweep_func(base_args.sweep)
    sweep_func(base_args)
