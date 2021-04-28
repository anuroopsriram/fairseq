from copy import deepcopy
from pathlib import Path

import numpy as np

import submit

wer_args = (
    '/datasets01/librispeech/021419/lm/4-gram.arpa',
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
    'mask-prob': 0.5,                       # 0.65
    'layerdrop': 0.1,                       # 0.05
    'mask-channel-selection': 'static',
    'mask-channel-other': 0,
    'mask-channel-length': 64,
    'mask-channel-prob': 0.5,
    'zero-infinity': True,
    'feature-grad-mult': 0.0,
    # 'freeze-finetune-updates': 10000,
    'freeze-finetune-updates': 0,           # 10000
    # 'validate-after-updates': 10000,
    'optimizer': 'adam',
    'adam-betas': (0.9, 0.98),
    'adam-eps': 1e-08,
    'lr': 2e-05,                            # 5e-5
    'lr-scheduler': 'tri_stage',
    # 'warmup-steps': 8000,                   # 0.1
    # 'hold-steps': 32000,                    # 0.4
    # 'decay-steps': 40000,                   # 0.5
    'final-lr-scale': 0.05,
    'final-dropout': 0.0,
    'dropout': 0.0,
    'activation-dropout': 0.1,
    'criterion': 'ctc',
    'attention-dropout': 0.0,
    # 'max-tokens': 3_200_000,                # 3200000
    'max-tokens': 4_800_000,                # 3200000
    'seed': 2337,                           # 1
    # 'log-format': 'json',
    'log-interval': 200,                    # 200
    'ddp-backend': 'no_c10d',               # legacy_ddp
    # 'validate-interval-updates': 500,       # 0
    # 'validate-interval': 10000,             # 50
    # 'save-interval-updates': 500,           # 10000
    'no-epoch-checkpoints': True,
}


@submit.register_sweep
def w2v_base(base_args):
    checkpoints = {
        # "ablation.baseline.ls960h.8x400.ft": [
        #     "logs/ablation.baseline.ls960h.8x400/ls960h.baselinelr0.0005.ls960h",
        # ],
        # "ablation.aug.ls10h.3x100.ft": [
        #     "logs/ablation.aug.ls10h.3x100/ls10h.add8.15lr0.0005.ls10h",
        #     "logs/ablation.aug.ls10h.3x100/ls10h.noauglr0.0005.ls10h",
        # ],
        "ablation.aug.50M.ls10h.3x100.ft": [
            # "logs/ablation.aug.50M.ls10h.3x100/ls10h.noauglr0.0005.ls10h",
            # "logs/ablation.aug.50M.ls10h.3x100/ls10h.add8.15lr0.0005.ls10h",
        ],
        # "ablation.aug.ls50h.3x200.ft": [
        #     "logs/ablation.aug.ls50h.3x150/ls50h.noauglr0.0005.ls50h",
        #     "logs/ablation.aug.ls50h.3x150/ls50h.add8.15lr0.0005.ls50h",
        # ],
        # "ablation.aug.ls100h.3x200.ft": [
        #     "logs/ablation.aug.ls100h.3x200/ls100h.noauglr0.0005.ls100h",
        #     "logs/ablation.aug.ls100h.3x200/ls100h.add8.15lr0.0005.ls100h",
        #     "logs/ablation.aug.ls100h.3x200/ls100h.sameaug.add8.15lr0.0005.ls100h",
        # ],
        # "ablation.aug.ls400h.3x300.ft": [
        #     "logs/ablation.aug.ls400h.3x300/ls400h.noauglr0.0005.ls400h",
        #     "logs/ablation.aug.ls400h.3x300/ls400h.add8.15lr0.0005.ls400h",
        # ],
        # "ablation.aug.ls960h.3x400.ft": [
        #     "logs/ablation.aug.ls960h.3x400/ls960h.noauglr0.0005.ls960h",
        #     "logs/ablation.aug.ls960h.3x400/ls960h.add8.15lr0.0005.ls960h",
        # ],
        # "ablation.mlp.ls960h.3x400.ft": [
        #     # "logs/ablation.mlp.ls960h.3x400/lr0.0005.cmlpFalse.tmlpTrue.bnFalse.actrelu.scale4.nhid0.unlab",
        #     # "logs/ablation.mlp.ls960h.3x400/lr0.0005.cmlpFalse.tmlpTrue.bnFalse.actrelu.scale4.nhid1.unlab",
        #     # "logs/ablation.mlp.ls960h.3x400/lr0.0005.cmlpFalse.tmlpTrue.bnFalse.actrelu.scale4.nhid2.unlab",
        #     # "logs/ablation.mlp.ls960h.3x400/lr0.0005.cmlpTrue.tmlpTrue.bnTrue.actrelu.scale4.nhid0.unlab",
        #     # "logs/ablation.mlp.ls960h.3x400/lr0.0005.cmlpTrue.tmlpTrue.bnTrue.actrelu.scale4.nhid1.unlab",
        #     # "logs/ablation.mlp.ls960h.3x400/lr0.0005.cmlpTrue.tmlpTrue.bnTrue.actrelu.scale4.nhid2.unlab",
        # ],
        "ablation.glu.ls960h.3x400.ft": [
        #     "logs/ablation.glu.ls960h.3x400/ls960h.geglu.lr0.0005.ls960h",
        #     "logs/ablation.glu.ls960h.3x400/ls960h.swiglu.lr0.0005.ls960h",

            # "logs/ablation.glu.ls960h.3x400/ls960h.geglu_bias.lr0.0005.ls960h",
            # "logs/ablation.glu.ls960h.3x400/ls960h.swiglu_bias.lr0.0005.ls960h",
        ],
        "ablation.lconv.ls960h.3x400.ft": [
            # "logs/ablation.lconv.ls960h.3x400/ls960h.dc_last2.lr0.0005.moddefault.ls960h",
            # "logs/ablation.lconv.ls960h.3x400/ls960h.dc_last2.lr0.0005.moddo0.1.ls960h",
            "logs/ablation.lconv.ls960h.3x400/ls960h.dc_last4.lr0.0005.moddefault.ls960h",
            "logs/ablation.lconv.ls960h.3x400/ls960h.dc_last4.lr0.0005.moddo0.1.ls960h",
            # "logs/ablation.lconv.ls960h.3x400/ls960h.lc_last2.lr0.0005.moddefault.ls960h",
            # "logs/ablation.lconv.ls960h.3x400/ls960h.lc_last2.lr0.0005.moddo0.1.ls960h",
            "logs/ablation.lconv.ls960h.3x400/ls960h.lc_last4.lr0.0005.moddefault.ls960h",
            "logs/ablation.lconv.ls960h.3x400/ls960h.lc_last4.lr0.0005.moddo0.1.ls960h",
        ],
        "ablation.conf.ls960h.3x400.ft": [
            # "logs/ablation.conf.ls960h.3x400/ls960h.conf.lr0.0005.ks3.normbatchnorm.ls960h",
            # "logs/ablation.conf.ls960h.3x400/ls960h.conf.lr0.0005.ks3.normlayernorm.ls960h",
            # "logs/ablation.conf.ls960h.3x400/ls960h.conf_rp.lr0.0005.ks3.normbatchnorm.ls960h",
            # "logs/ablation.conf.ls960h.3x400/ls960h.conf_rp.lr0.0005.ks3.normlayernorm.ls960h",
        ],
    }
    run_args_list = {
        "lab.1h": dict(updates=20_000, nodes=1, gpus=1, update_freq=4),
        # "lab.10h": dict(updates=80_000, nodes=1, gpus=8, update_freq=1),
        "lab.100h": dict(updates=100_000, nodes=1, gpus=1, update_freq=4),
    }

    default_params = {
        "lab.1h": {
            "validate-interval": 500,
            "save-interval": 100000,
            "save-interval-updates": 500,
            "keep-interval-updates": 1,
            "validate-after-updates": 5000,

            "mask-channel-prob": 0.25,
            "freeze-finetune-updates": 10_000,
        },
        "lab.10h": {
        },
        "lab.100h": {
            "validate-interval": 2000,
            "save-interval": 100000,
            "save-interval-updates": 2000,
            "keep-interval-updates": 1,
            "validate-after-updates": 10_000,
        }
    }

    mask_lens = [10]
    mask_probs = [
        0.45,
        0.65,
    ]
    dos = [0.1]
    lrs = [
        3e-05,
        5e-05,
    ]

    for name, checkpoints_list in checkpoints.items():
        for checkpoint in checkpoints_list:
            for dset, run_params in run_args_list.items():
                checkpoint = Path(checkpoint)
                args = deepcopy(base_args)
                args.nodes = run_params["nodes"]
                args.gpus = run_params["gpus"]
                args.name = (args.name or name) + "/" + checkpoint.name

                param_sweeps = [
                    (
                        f"{dset}.lr{lr}.mlen{mlen}.mprob{mprob}.do{do}",
                        {
                            # Checkpoint
                            "w2v-path": checkpoint / "checkpoint_best.pt",
                            # N-Gram
                            'wer-args': wer_args,

                            "lr": lr,
                            'mask-length': mlen,
                            'mask-prob': mprob,
                            "max-update": run_params["updates"],
                            "update-freq": run_params["update_freq"],
                            "phase-ratio": [0.1, 0.4, 0.5],

                            "augment-audio": False,
                            "layerdrop": do,
                            "activation-dropout": do,
                            "freeze-finetune-updates": 0,

                            **default_params[dset]
                        }
                    )
                    for do in dos
                    for mlen in mask_lens
                    for mprob in mask_probs
                    for lr in lrs
                ]
                submit.run_sweeps(args, base_params, param_sweeps, dataset=dset, skip_if_cp_exists=False)


@submit.register_sweep
def w2v_scratch(base_args):
    run_args_list = {
        # "lab.1h": dict(updates=20_000, nodes=1, gpus=1, update_freq=4),
        # "lab.10h": dict(updates=80_000, nodes=1, gpus=8, update_freq=1),
        "lab.100h": dict(updates=100_000, nodes=1, gpus=1, update_freq=4),
    }
    default_params = {
        "lab.1h": {
            "validate-interval": 500,
            "save-interval": 100000,
            "save-interval-updates": 500,
            "keep-interval-updates": 1,
            "validate-after-updates": 5000,

            "mask-channel-prob": 0.25,
            "freeze-finetune-updates": 10_000,
        },
        "lab.100h": {
            "validate-interval": 2000,
            "save-interval": 100000,
            "save-interval-updates": 2000,
            "keep-interval-updates": 1,
            "validate-after-updates": 10_000,
        }
    }
    mask_lens = [10]
    mask_probs = [
        0.45,
        0.65,
    ]
    dos = [0.1]
    lrs = [
        3e-04,
        3e-03,
    ]
    checkpoint = Path("logs/ablation.baseline.ls960h.8x400/ls960h.baselinelr0.0005.ls960h")
    name = "logs/ablation.scratch"
    for dset, run_params in run_args_list.items():
        args = deepcopy(base_args)
        args.nodes = run_params["nodes"]
        args.gpus = run_params["gpus"]
        args.name = (args.name or name)
        param_sweeps = [
            (
                f"{dset}.lr{lr}.mlen{mlen}.mprob{mprob}.do{do}",
                {
                    # Checkpoint
                    "w2v-path": checkpoint / "checkpoint_best.pt",
                    # N-Gram
                    'wer-args': wer_args,

                    "lr": lr,
                    'mask-length': mlen,
                    'mask-prob': mprob,
                    "max-update": run_params["updates"],
                    "update-freq": run_params["update_freq"],
                    "phase-ratio": [0.1, 0.4, 0.5],

                    "augment-audio": False,
                    "layerdrop": do,
                    "activation-dropout": do,
                    "freeze-finetune-updates": 0,
                    "feature-grad-mult": 1.,
                    "no-pretrained-weights": True,

                    **default_params[dset]
                }
            )
            for do in dos
            for mlen in mask_lens
            for mprob in mask_probs
            for lr in lrs
        ]
        submit.run_sweeps(args, base_params, param_sweeps, dataset=dset, skip_if_cp_exists=False)


if __name__ == '__main__':
    parser = submit.create_parser()
    base_args = parser.parse_args()
    sweep_func = submit.get_sweep_func(base_args.sweep)
    sweep_func(base_args)
