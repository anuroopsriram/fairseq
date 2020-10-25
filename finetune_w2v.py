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


def w2v_base_s2s_250k(args, params):
    max_update = 320000

    params.update({
        'labels': '10k',
        'arch': 'wav2vec_seq2seq',
        'find-unused-parameters': True,
        'layerdrop': 0.2,
        'decoder-layerdrop': 0.25,
        'mask-channel-prob': 0.3,

        'decoder-layers': 3,  # TODO
        'decoder-embed-dim': 1024,
        'decoder-ffn-embed-dim': 4096,
        'decoder-attention-heads': 16,

        'freeze-finetune-updates': 0,
        'validate-after-updates': 0,

        'max-update': max_update,
        'warmup-steps': int(max_update * 0.1),
        'hold-steps': int(max_update * 0.4),
        'decay-steps': int(max_update * 0.5),
        'final-lr-scale': 0.05,

        'dropout': 0.3,
        'activation-dropout': 0.1,
        'attention-dropout': 0.1,

        'criterion': 'cross_entropy',
    })
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


def w2v_conformer_relpos_large(args, params):
    args.name = args.name or 'w2v.conformer.relpos.600k.16nd.ft'
    args.nodes = 3
    return args, params


def w2v_conformer_400k_lm(args, params):
    args.name = args.name or 'w2v.conformer.400k.ft.4glm'
    args.nodes = 3
    return args, params


def w2v_conformer_relpos_s2s(args, params):
    max_update = 320000

    args.name = args.name or 'w2v.conformer.relpos.s2s.400k.ft'
    args.nodes = 3

    params.update({
        'labels': '10k',
        'arch': 'wav2vec_seq2seq',
        'find-unused-parameters': True,
        'layerdrop': 0.1,
        'mask-channel-prob': 0.3,

        # 'decoder-layers': 3,
        # 'decoder-embed-dim': 1024,
        # 'decoder-ffn-embed-dim': 4096,
        # 'decoder-attention-heads': 16,

        'decoder-layers': 2,
        'decoder-embed-dim': 640,
        'decoder-ffn-embed-dim': 1024,
        'decoder-attention-heads': 8,

        'freeze-finetune-updates': 10000,
        'validate-after-updates': 10000,

        'max-update': max_update,
        'warmup-steps': int(max_update * 0.2),
        'hold-steps': int(max_update * 0.4),
        'decay-steps': int(max_update * 0.4),
        'final-lr-scale': 0.05,

        'decoder-layerdrop': 0.1,
        'decoder-dropout': 0.2,
        'decoder-activation-dropout': 0.2,
        'decoder-attention-dropout': 0.2,

        'criterion': 'cross_entropy',

        'share-decoder-input-output-embed': False,

        'log-interval': 500,
        'eval-wer': True,
    })
    del params['zero-infinity']
    del params['post-process']
    # del params['best-checkpoint-metric']

    return args, params


def sup_conformer_relpos_s2s(args, params):
    max_update = 800000

    args.name = args.name or 'sup.conformer.relpos.s2s'
    args.nodes = 3

    params.update({
        'labels': '10k',
        'arch': 'wav2vec_seq2seq',
        'find-unused-parameters': True,
        'layerdrop': 0.2,
        'decoder-layerdrop': 0.25,
        'mask-channel-prob': 0.3,

        'decoder-layers': 3,  # TODO
        'decoder-embed-dim': 1024,
        'decoder-ffn-embed-dim': 4096,
        'decoder-attention-heads': 16,

        'freeze-finetune-updates': 0,
        'validate-after-updates': 0,

        'max-update': max_update,
        'warmup-steps': int(max_update * 0.2),
        'hold-steps': int(max_update * 0.4),
        'decay-steps': int(max_update * 0.4),
        'final-lr-scale': 0.05,

        'decoder-dropout': 0.3,
        'decoder-activation-dropout': 0.1,
        'decoder-attention-dropout': 0.1,

        'criterion': 'cross_entropy',

        'share-decoder-input-output-embed': False,

        'no-pretrained-weights': True,
        'eval-wer': True,
    })
    del params['zero-infinity']
    del params['post-process']
    # del params['best-checkpoint-metric']

    return args, params


def sup_conformer_relpos_800k(args, params):
    args.name = args.name or 'sup.conformer.relpos.250k'
    args.nodes = 3
    args.no32gb = True
    params.update({
        'no-pretrained-weights': True,
        'apply-mask': True,
        'max-update': 800000,

        'warmup-steps': 40000,
        'hold-steps': 400000,
        'decay-steps': 360000,

        'freeze-finetune-updates': 0,
        'validate-after-updates': 0,
    })
    return args, params


# def sup_conformer_relpos_250k_lm(args, params):
#     args.name = args.name or 'sup.conformer.relpos.250k.4glm'
#     args.nodes = 3
#     params.update({
#         'no-pretrained-weights': True,
#         'apply-mask': True,
#     })
#     return args, params
#
#
# def sup_conformer_relpos_logmel_250k(args, params):
#     args.name = args.name or 'sup.conformer.relpos.250k'
#     args.nodes = 4
#     params.update({
#         'apply-mask': False,
#         'no-pretrained-weights': True,
#         # 'logmel': True,
#         # 'in-d': 80,
#         # 'specaug-prob': 0.8,
#         # 'conv-feature-layers': [(512, 7, 2)] * 2,
#     })
#     return args, params


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
def sweep_w2v_conformer_relpos_large_10h(base_args):
    # lrs = [6e-05, 2e-05]
    # lrs = [1e-04]
    lrs = [5e-05]
    checkpoints = [
        Path('logs/w2v.conformer.relpos.600K.16nd/lr0.001.unlab'),
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
    submit.run_sweeps(w2v_conformer_relpos_large, base_args, base_params, param_sweeps, dataset='lab.10h')


@submit.register_sweep
def sweep_w2v_conformer_relpos_large_960h(base_args):
    # lrs = [6e-05, 2e-05]
    # lrs = [1e-04]
    lrs = [5e-05]
    checkpoints = [
        Path('logs/w2v.conformer.relpos.600K.16nd/lr0.001.unlab'),
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
    submit.run_sweeps(w2v_conformer_relpos_large, base_args, base_params, param_sweeps, dataset='lab.960h')


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
    # lrs = [2e-05]
    lrs = [6e-05, 6e-06]
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


@submit.register_sweep
def sweep_w2v_conformer_relpos_s2s_960h(base_args):
    lrs = [1e-04]
    checkpoints = [
        Path('logs/w2v.conformer.relpos.400k/dim512.enclyrs17.lr0.0005.rpemb16.unlab'),
    ]
    declyrs = [2, 3]
    lds = [0.15]
    dos = [0.3]
    decparams = [
        # (640, 1024, 8),
        (512, 768, 8)
    ]
    param_sweeps = [
        (
            f'{checkpoint.name}/lr{lr}.declyrs{declyr}.ld{ld}.do{do}.ddim{decdim}.dffndim{decffndim}.dhd{dechead}',
            {
                'decoder-layerdrop': ld,
                'decoder-dropout': do,
                'decoder-activation-dropout': do,
                'decoder-attention-dropout': do,

                'decoder-embed-dim': decdim,
                'decoder-ffn-embed-dim': decffndim,
                'decoder-attention-heads': dechead,

                'w2v-path': checkpoint / 'checkpoint_best.pt',
                'lr': lr,
                "eval-wer": True,
                "decoder-layers": declyr,
            },
        )
        for checkpoint in checkpoints
        for lr in lrs
        for declyr in declyrs
        for ld in lds
        for do in dos
        for (decdim, decffndim, dechead) in decparams
    ]
    submit.run_sweeps(w2v_conformer_relpos_s2s, base_args, base_params, param_sweeps, dataset='lab.960h')
    # submit.run_sweeps(w2v_conformer_relpos_s2s, base_args, base_params, param_sweeps, dataset='lab.10h')


@submit.register_sweep
def sweep_sup_conformer_relpos_s2s_960h(base_args):
    lrs = [1e-04]
    checkpoints = [
        Path('logs/w2v.conformer.relpos.400k/dim512.enclyrs17.lr0.0005.rpemb16.unlab'),
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
    submit.run_sweeps(sup_conformer_relpos_s2s, base_args, base_params, param_sweeps, dataset='lab.960h')


@submit.register_sweep
def sweep_sup_conformer_relpos_800k_960h(base_args):
    # lrs = [1e-2, 1e-3, 1e-4]
    lrs = [3e-4, 3e-5]

    clips = [500]
    checkpoints = [
        Path('logs/w2v.conformer.relpos.250k/dim512.enclyrs17.lr0.001.rpemb16'),
    ]
    param_sweeps = [
        (
            f'{checkpoint.name}/lr{lr}.cn{clip}',
            {
                'w2v-path': checkpoint / 'checkpoint_best.pt',
                'lr': lr,
                'clip-norm': clip,
            },
        )
        for clip in clips
        for checkpoint in checkpoints
        for lr in lrs
    ]
    submit.run_sweeps(sup_conformer_relpos_800k, base_args, base_params, param_sweeps, dataset='lab.960h')


@submit.register_sweep
def sweep_sup_conformer_relpos_800k_960h_4glm(base_args):
    # lrs = [1e-2, 1e-3, 1e-4]
    # lrs = [3e-4, 6e-4]
    lrs = [1e-4]
    clips = [200]
    checkpoints = [
        Path('logs/w2v.conformer.relpos.250k/dim512.enclyrs17.lr0.001.rpemb16'),
    ]
    param_sweeps = [
        (
            f'{checkpoint.name}/lr{lr}.cn{clip}',
            {
                'w2v-path': checkpoint / 'checkpoint_best.pt',
                'lr': lr,
                'clip-norm': clip * 2,
                'wer-args': wer_args,
            },
        )
        for clip in clips
        for checkpoint in checkpoints
        for lr in lrs
    ]
    submit.run_sweeps(w2v_base_400k, base_args, base_params, param_sweeps, dataset='lab.960h')


@submit.register_sweep
def sweep_w2v_conformer_partial_relpos_250k_17lyrs(base_args):
    checkpoints = {
        # 'none': Path('logs/w2v.conformer.relpos.2x100k/dim512.enclyrs8.lr0.001.rpemb16.mhanone.unlab/'),
        # 'all': Path('logs/w2v.conformer.relpos.2x100k/dim512.enclyrs8.lr0.001.rpemb16.mhaall.unlab/'),
        # 'first4': Path('logs/w2v.conformer.relpos.2x100k/dim512.enclyrs8.lr0.001.rpemb16.mhafirst4.unlab/'),
        # 'last4': Path('logs/w2v.conformer.relpos.2x100k/dim512.enclyrs8.lr0.001.rpemb16.mhalast4.unlab/'),
        # 'alt': Path('logs/w2v.conformer.relpos.2x100k/dim512.enclyrs8.lr0.001.rpemb16.mhaalt.unlab/'),

        'alt': Path('logs/w2v.conformer.relpos.250k/dim512.enclyrs17.lr0.001.rpemb16.mhaalt.unlab/'),
        'first4': Path('logs/w2v.conformer.relpos.250k/dim512.enclyrs17.lr0.001.rpemb16.mhafirst4.unlab/'),
        'last4': Path('logs/w2v.conformer.relpos.250k/dim512.enclyrs17.lr0.001.rpemb16.mhalast4.unlab/'),
        'first8': Path('logs/w2v.conformer.relpos.250k/dim512.enclyrs17.lr0.001.rpemb16.mhafirst8.unlab/'),
        'last8': Path('logs/w2v.conformer.relpos.250k/dim512.enclyrs17.lr0.001.rpemb16.mhalast8.unlab/'),
    }
    lrs = [2e-5]
    param_sweeps = [
        (
            f'{checkpoint.name}/{name}.lr{lr}',
            {
                'w2v-path': checkpoint / 'checkpoint_best.pt',
                'lr': lr,
            },
        )
        for name, checkpoint in checkpoints.items()
        for lr in lrs
    ]
    # base_args.name = 'w2v.conformer.relpos.2x100k.ft'
    base_args.name = 'w2v.conformer.relpos.250k.ft'
    submit.run_sweeps(w2v_base_400k, base_args, base_params, param_sweeps, dataset='lab.10h')


def sup_tmp(args, params):
    max_update = 800000

    args.name = args.name or 'sup.conformer.relpos.s2s'
    args.nodes = 3

    params.update({
        'labels': '10k',
        'arch': 'wav2vec_seq2seq',
        'find-unused-parameters': True,
        'layerdrop': 0.2,
        'decoder-layerdrop': 0.25,
        'mask-channel-prob': 0.3,

        'decoder-layers': 3,  # TODO
        'decoder-embed-dim': 1024,
        'decoder-ffn-embed-dim': 4096,
        'decoder-attention-heads': 16,

        'freeze-finetune-updates': 0,

        'max-update': max_update,
        'warmup-steps': int(max_update * 0.2),
        'hold-steps': int(max_update * 0.4),
        'decay-steps': int(max_update * 0.4),
        'final-lr-scale': 0.05,

        'decoder-dropout': 0.3,
        'decoder-activation-dropout': 0.1,
        'decoder-attention-dropout': 0.1,

        'criterion': 'cross_entropy',

        'share-decoder-input-output-embed': False,

        'no-pretrained-weights': True,

        'validate-after-updates': 1,
        "eval-wer": True,
    })
    del params['zero-infinity']
    del params['post-process']
    del params['best-checkpoint-metric']

    return args, params


@submit.register_sweep
def sweep_sup_s2s_tmp(base_args):
    lrs = [1e-04]
    checkpoints = [
        Path('logs/w2v.base.250k/dim704.enclyrs17.lr0.0001/'),
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
    submit.run_sweeps(sup_tmp, base_args, base_params, param_sweeps, dataset='lab.10h')


if __name__ == '__main__':
    parser = submit.create_parser()
    base_args = parser.parse_args()
    sweep_func = submit.get_sweep_func(base_args.sweep)
    sweep_func(base_args)
